use chrono::prelude::*;
use clap::{Arg, ArgAction, Command};
use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use shaderc::{Compiler, ShaderKind};
use std::collections::HashMap;
use std::fs;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::mpsc::channel;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalSize,
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(non_snake_case)]
struct Uniforms {
    iResolution: [f32; 3],
    _padding1: f32, // Padding to align to 16 bytes
    iTime: f32,
    iTimeDelta: f32,
    iFrame: i32,
    iFrameRate: f32,
    iChannelTime: [f32; 4],
    iMouse: [f32; 4],
    iDate: [i32; 4],
}

impl Uniforms {
    fn new(width: f32, height: f32) -> Self {
        Self {
            iResolution: [width, height, 1.0],
            _padding1: 0.0,
            iTime: 0.0,
            iTimeDelta: 0.0,
            iFrame: 0,
            iFrameRate: 0.0,
            iChannelTime: [0.0; 4],
            iMouse: [0.0; 4],
            iDate: [0; 4],
        }
    }
}

struct State {
    _instance: wgpu::Instance,
    surface: Option<wgpu::Surface>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: Option<wgpu::SurfaceConfiguration>,
    size: PhysicalSize<u32>,

    render_pipeline: wgpu::RenderPipeline,
    pipeline_layout: wgpu::PipelineLayout,
    vertex_shader_module: wgpu::ShaderModule,
    start_time: Instant,
    previous_time: f32,
    frame_count: u32,
    _time: f32,
    mouse_pos: [f32; 2],
    mouse_pressed: bool,

    shader_module: wgpu::ShaderModule,
    shader_path: PathBuf,
    defines: HashMap<String, String>,

    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    uniforms: Uniforms,

    output_file: Option<PathBuf>,
    render_texture: Option<wgpu::Texture>,
}

impl State {
    async fn new(
        window: Option<&winit::window::Window>,
        size: PhysicalSize<u32>,
        shader_path: PathBuf,
        defines: HashMap<String, String>,
        output_file: Option<PathBuf>,
    ) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });
        let surface = if let Some(window) = window {
            Some(unsafe { instance.create_surface(window).unwrap() })
        } else {
            None
        };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface.as_ref(),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find an appropriate adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let config = if let Some(surface) = &surface {
            let surface_caps = surface.get_capabilities(&adapter);
            let surface_format = surface_caps.formats[0];
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: surface_caps.alpha_modes[0],
                view_formats: vec![surface_format],
            };
            surface.configure(&device, &config);
            Some(config)
        } else {
            None
        };

        // Compile shaders
        let (vertex_shader_module, fragment_shader_module) =
            Self::compile_shaders(&device, &shader_path, &defines);

        // Uniforms
        let uniforms = Uniforms::new(size.width as f32, size.height as f32);

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Uniform Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&uniform_bind_group_layout],
            push_constant_ranges: &[],
        });

        let format = if let Some(config) = &config {
            config.format
        } else {
            wgpu::TextureFormat::Bgra8UnormSrgb
        };

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_shader_module,
                entry_point: "main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &fragment_shader_module,
                entry_point: "main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let render_texture = if output_file.is_some() {
            Some(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Render Texture"),
                size: wgpu::Extent3d {
                    width: size.width,
                    height: size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            }))
        } else {
            None
        };

        Self {
            _instance: instance,
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            pipeline_layout,
            vertex_shader_module,
            start_time: Instant::now(),
            _time: 0.0,
            mouse_pos: [0.0; 2],
            mouse_pressed: false,
            shader_module: fragment_shader_module,
            shader_path,
            defines,
            uniform_buffer,
            uniform_bind_group,
            uniforms,
            output_file,
            render_texture,
            previous_time: 0.0,
            frame_count: 0,
        }
    }

    fn apply_defines(source: &str, defines: &HashMap<String, String>) -> String {
        let mut result = String::new();
        for (key, value) in defines {
            if value.is_empty() {
                result.push_str(&format!("#define {}\n", key));
            } else {
                result.push_str(&format!("#define {} {}\n", key, value));
            }
        }
        result.push_str(source);
        result
    }

    fn inject_shader_preamble(source: &str) -> String {
        let preamble = r#"#version 450

    layout(std140, set = 0, binding = 0) uniform Uniforms {
        vec3 iResolution;
        float _padding1;
        float iTime;
        float iTimeDelta;
        int iFrame;
        float iFrameRate;
        vec4 iChannelTime;
        vec4 iMouse;
        ivec4 iDate;
    };

    layout(location = 0) out vec4 FragColor;

    "#;

        let postamble = r#"

    void main() {
        vec2 fragCoord = gl_FragCoord.xy;
        fragCoord.y = iResolution.y - fragCoord.y; // Flip the Y-coordinate
        mainImage(FragColor, fragCoord);
    }

    "#;
        format!("{}{}{}", preamble, source, postamble)
    }

    fn compile_shaders(
        device: &wgpu::Device,
        shader_path: &PathBuf,
        defines: &HashMap<String, String>,
    ) -> (wgpu::ShaderModule, wgpu::ShaderModule) {
        let mut compiler = Compiler::new().unwrap();

        // Vertex Shader
        let vertex_shader_source = r#"
            #version 450
            void main() {
                vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
                gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
            }
            "#;

        let vertex_spirv = compiler
            .compile_into_spirv(
                vertex_shader_source,
                ShaderKind::Vertex,
                "vertex.glsl",
                "main",
                None,
            )
            .expect("Failed to compile vertex shader");

        let vertex_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vertex Shader"),
            source: wgpu::util::make_spirv(vertex_spirv.as_binary_u8()),
        });

        // Fragment Shader
        let shader_source =
            fs::read_to_string(shader_path).expect("Failed to read shader source file");
        let shader_source = Self::apply_defines(&shader_source, defines);
        let shader_source = Self::inject_shader_preamble(&shader_source);

        let mut compile_options = shaderc::CompileOptions::new().unwrap();
        compile_options.set_target_env(
            shaderc::TargetEnv::Vulkan,
            shaderc::EnvVersion::Vulkan1_2 as u32,
        );

        let fragment_spirv = match compiler.compile_into_spirv(
            &shader_source,
            ShaderKind::Fragment,
            shader_path.to_str().unwrap(),
            "main",
            Some(&compile_options),
        ) {
            Ok(binary) => binary,
            Err(e) => {
                // Print error message and shader source with line numbers
                eprintln!("Failed to compile fragment shader: {}", e);
                eprintln!("Transformed shader with line numbers:");
                for (i, line) in shader_source.lines().enumerate() {
                    eprintln!("{:4}: {}", i + 1, line);
                }
                panic!("Shader compilation failed");
            }
        };

        let fragment_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fragment Shader"),
            source: wgpu::util::make_spirv(fragment_spirv.as_binary_u8()),
        });

        (vertex_shader_module, fragment_shader_module)
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.size = new_size;
        if let Some(surface) = &self.surface {
            self.config.as_mut().unwrap().width = new_size.width;
            self.config.as_mut().unwrap().height = new_size.height;
            surface.configure(&self.device, self.config.as_ref().unwrap());
        }
        self.uniforms.iResolution = [new_size.width as f32, new_size.height as f32, 1.0];
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_pos = [position.x as f32, position.y as f32];
                false
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if *button == MouseButton::Left {
                    self.mouse_pressed = *state == ElementState::Pressed;
                }
                false
            }
            _ => false,
        }
    }

    fn update(&mut self) {
        let current_time = self.start_time.elapsed().as_secs_f32();
        self.uniforms.iTimeDelta = current_time - self.previous_time;
        if self.uniforms.iTimeDelta <= 0.0 {
            self.uniforms.iTimeDelta = 1.0 / 60.0; // Default to 60 FPS if delta is zero or negative
        }
        self.previous_time = current_time;

        self.uniforms.iTime = current_time;
        self.uniforms.iFrame = self.frame_count as i32;
        self.frame_count += 1;

        self.uniforms.iFrameRate = 1.0 / self.uniforms.iTimeDelta;

        // Update iMouse
        self.uniforms.iMouse = [
            self.mouse_pos[0],
            self.mouse_pos[1],
            if self.mouse_pressed {
                self.mouse_pos[0]
            } else {
                0.0
            },
            if self.mouse_pressed {
                self.mouse_pos[1]
            } else {
                0.0
            },
        ];

        // Update iDate
        let now = chrono::Local::now();
        let year = now.year();
        let month = now.month() as i32;
        let day = now.day() as i32;
        let seconds = now.hour() * 3600 + now.minute() * 60 + now.second();
        self.uniforms.iDate = [year, month, day, seconds as i32];

        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output_texture = if let Some(surface) = &self.surface {
            Some(surface.get_current_texture()?)
        } else {
            None
        };

        let view = if let Some(output_texture) = &output_texture {
            output_texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default())
        } else {
            self.render_texture
                .as_ref()
                .unwrap()
                .create_view(&wgpu::TextureViewDescriptor::default())
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        if let Some(output_texture) = output_texture {
            output_texture.present();
        }

        Ok(())
    }

    fn render_to_file(&mut self) {
        self.update();
        self.render().unwrap();

        let bytes_per_pixel = 4; // RGBA8 format
        let unpadded_bytes_per_row = bytes_per_pixel * self.size.width;
        let padded_bytes_per_row = ((unpadded_bytes_per_row + 255) / 256) * 256; // Align to 256 bytes

        let buffer_size = padded_bytes_per_row as u64 * self.size.height as u64;

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Read Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: self.render_texture.as_ref().unwrap(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(NonZeroU32::new(padded_bytes_per_row).unwrap()),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: self.size.width,
                height: self.size.height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        futures::executor::block_on(receiver).unwrap().unwrap();

        let padded_data = buffer_slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((unpadded_bytes_per_row * self.size.height) as usize);

        // Extract the actual image data, excluding padding
        for chunk in padded_data.chunks(padded_bytes_per_row as usize) {
            pixels.extend_from_slice(&chunk[..unpadded_bytes_per_row as usize]);
        }

        let image_buffer = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
            self.size.width,
            self.size.height,
            pixels,
        )
        .expect("Failed to create image buffer");

        image_buffer
            .save(self.output_file.as_ref().unwrap())
            .expect("Failed to save image");

        println!("Image saved to {:?}", self.output_file.as_ref().unwrap());
    }

    fn recreate_render_pipeline(&mut self) -> bool {
        if let Some(fragment_shader_module) =
            self.compile_fragment_shader(&self.device, &self.shader_path, &self.defines)
        {
            self.shader_module = fragment_shader_module;

            self.render_pipeline =
                self.device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some("Render Pipeline"),
                        layout: Some(&self.pipeline_layout),
                        vertex: wgpu::VertexState {
                            module: &self.vertex_shader_module,
                            entry_point: "main",
                            buffers: &[],
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &self.shader_module,
                            entry_point: "main",
                            targets: &[Some(wgpu::ColorTargetState {
                                format: self.config.as_ref().unwrap().format,
                                blend: Some(wgpu::BlendState::REPLACE),
                                write_mask: wgpu::ColorWrites::ALL,
                            })],
                        }),
                        primitive: wgpu::PrimitiveState::default(),
                        depth_stencil: None,
                        multisample: wgpu::MultisampleState::default(),
                        multiview: None,
                    });
            true // Indicate success
        } else {
            // Compilation failed; keep using the existing pipeline
            false // Indicate failure
        }
    }

    fn compile_fragment_shader(
        &self,
        device: &wgpu::Device,
        shader_path: &PathBuf,
        defines: &HashMap<String, String>,
    ) -> Option<wgpu::ShaderModule> {
        let mut compiler = Compiler::new().unwrap();

        println!("Compiling shader: {}", shader_path.display());
        let shader_source =
            fs::read_to_string(shader_path).expect("Failed to read shader source file");
        let shader_source = Self::apply_defines(&shader_source, defines);
        let shader_source = Self::inject_shader_preamble(&shader_source);

        let mut compile_options = shaderc::CompileOptions::new().unwrap();
        compile_options.set_target_env(
            shaderc::TargetEnv::Vulkan,
            shaderc::EnvVersion::Vulkan1_2 as u32,
        );

        let fragment_spirv = match compiler.compile_into_spirv(
            &shader_source,
            ShaderKind::Fragment,
            shader_path.to_str().unwrap(),
            "main",
            Some(&compile_options),
        ) {
            Ok(binary) => binary,
            Err(e) => {
                eprintln!("Failed to compile fragment shader: {}", e);
                eprintln!("Transformed shader with line numbers:");
                for (i, line) in shader_source.lines().enumerate() {
                    eprintln!("{:4}: {}", i + 1, line);
                }
                return None; // Indicate failure
            }
        };

        Some(device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fragment Shader"),
            source: wgpu::util::make_spirv(fragment_spirv.as_binary_u8()),
        }))
    }
}

fn main() {
    let matches = Command::new("Shadertoy Harness")
        .version("0.1")
        .author("Steven Lovegrove")
        .about("Runs a Shadertoy shader in a harness")
        .arg(
            Arg::new("shader")
                .help("Input shader file")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .help("Render output to file instead of displaying GUI")
                .action(ArgAction::Set)
                .value_name("FILE"),
        )
        .arg(
            Arg::new("width")
                .long("width")
                .help("Width of the output image")
                .action(ArgAction::Set)
                .default_value("800"),
        )
        .arg(
            Arg::new("height")
                .long("height")
                .help("Height of the output image")
                .action(ArgAction::Set)
                .default_value("600"),
        )
        .arg(
            Arg::new("start_time")
                .long("start-time")
                .help("Override animation start time in seconds")
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("define")
                .long("define")
                .short('D')
                .help("Provide #define substitutions")
                .action(ArgAction::Append)
                .num_args(1)
                .value_name("DEFINE"),
        )
        .get_matches();

    let shader_path = PathBuf::from(matches.get_one::<String>("shader").unwrap());

    let output_file = matches
        .get_one::<String>("output")
        .map(|s| PathBuf::from(s));

    let width: u32 = matches
        .get_one::<String>("width")
        .unwrap()
        .parse()
        .expect("Invalid width");

    let height: u32 = matches
        .get_one::<String>("height")
        .unwrap()
        .parse()
        .expect("Invalid height");

    let start_time: Option<f32> = matches
        .get_one::<String>("start_time")
        .map(|s| s.parse().expect("Invalid start time"));

    let defines: HashMap<String, String> = matches
        .get_many::<String>("define")
        .unwrap_or_default()
        .map(|s| {
            let parts: Vec<&str> = s.split('=').collect();
            if parts.len() == 2 {
                (parts[0].to_string(), parts[1].to_string())
            } else if parts.len() == 1 {
                (parts[0].to_string(), String::new())
            } else {
                panic!("Invalid define: {}", s);
            }
        })
        .collect();

    if output_file.is_none() {
        // GUI mode
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_title("Shadertoy Harness")
            .with_inner_size(PhysicalSize::new(width, height))
            .build(&event_loop)
            .unwrap();

        let mut state = pollster::block_on(State::new(
            Some(&window),
            window.inner_size(),
            shader_path.clone(),
            defines,
            None,
        ));

        // Set start time
        if let Some(start_time) = start_time {
            state.start_time = Instant::now() - Duration::from_secs_f32(start_time);
        }

        // Watch shader file
        let (tx, rx) = channel();
        let mut watcher =
            RecommendedWatcher::new(move |res| tx.send(res).unwrap(), notify::Config::default())
                .unwrap();
        watcher
            .watch(&shader_path, RecursiveMode::NonRecursive)
            .unwrap();

        event_loop.run(move |event, _, control_flow| {
            match event {
                winit::event::Event::WindowEvent {
                    ref event,
                    window_id,
                } if window_id == window.id() => {
                    if !state.input(event) {
                        match event {
                            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                            WindowEvent::Resized(physical_size) => {
                                state.resize(*physical_size);
                            }
                            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                                state.resize(**new_inner_size);
                            }
                            _ => {}
                        }
                    }
                }
                winit::event::Event::RedrawRequested(_) => {
                    state.update();
                    match state.render() {
                        Ok(_) => {}
                        // Reconfigure the surface if lost
                        Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                        // The system is out of memory, we should probably quit
                        Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                        // All other errors (Outdated, Timeout) should be resolved by the next frame
                        Err(e) => eprintln!("{:?}", e),
                    }
                }
                winit::event::Event::MainEventsCleared => {
                    // Check for shader file changes
                    while let Ok(Ok(event)) = rx.try_recv() {
                        if matches!(event.kind, EventKind::Modify(_)) {
                            println!("Shader file changed, recompiling...");
                            if state.recreate_render_pipeline() {
                                println!("Shader recompiled successfully.");
                            } else {
                                println!("Shader recompilation failed. Using previous shader.");
                            }
                        }
                    }
                    window.request_redraw();
                }
                _ => {}
            }
        });
    } else {
        // Render to file mode
        let mut state = pollster::block_on(State::new(
            None,
            PhysicalSize::new(width, height),
            shader_path.clone(),
            defines,
            output_file.clone(),
        ));

        // Set start time
        if let Some(start_time) = start_time {
            state.start_time = Instant::now() - Duration::from_secs_f32(start_time);
        }

        // Update and render once
        state.render_to_file();
    }
}
