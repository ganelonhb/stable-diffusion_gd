# stable-diffusion.gd
A C++ module for Godot to interact with Stable-Diffusion.cpp.

## Todo

☐ URGENT A thread for the execution of both context loading and image generation<br>
☐ URGENT Actual Windows and Mac build support...<br>
☐ Support for img2img<br>
☐ Support for loading model from user::/<br>
☐ Improved API access to stable-diffusion.cpp for custom txt2img implementation<br>
☐ Custom GGML backend for loading from buffer<br>
☐ Suport for loading model from res::/

☐ Volk support...?

## Building

To get up-and-running, you will want to clone the code onto your machine, then follow these steps.

1. cd into the stable_diffusion_gd directory.
2. Compile stable diffusion with a backend of your choice (See [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)) (as of now, vulkan support is borked thanks to Godot's use of Volk).
3. Find both libstable-diffusion and libggml, and drop them into the proper directory in libs/
4. Copy the directory into your local copy of godot/modules/
5. Build Godot using Scons like you would any other module!

## Usage

stable-diffusion.gd offers robust interaction with Stable-Diffusion.cpp. Simply add the "StableDiffusion" node to a scene to make use of it. The next sections will be dedicated to teaching you how to use the node.

### The StableDiffusion Node -- Properties

The following properties can be used in GDScript using getters and setters that follow the get_property_name_here and set_property_name_here format, or set in the editor. The following section will explain the usage of each one.

**Prompt.** The prompt property sets the prompt to use with inference.

**Negative Prompt.** The negative prompt property specifies tokens to penalize, making it less likely that certain concepts will appear in the generation.

**Model Path.** The model path declares a path to seek the model in. Currently, it does *not* support Resources, nor the user:// notation. Unless GGML is updated to support loading models from a buffer, resources may not be supported for a while, but soon the codebase will be reorganized to be smarter and interact better with Godot.

**Width and Height.** The Width and Height properties set the width and height in pixels of the generated image.

**Sample Steps.** The sample steps parameter determines how many iterations of diffusion the image will go through. The number of steps you should use depend on the model you use. In general, for Stable Diffusion 1.5 models, just use the default setting.

**CFG Scale.** You can think of context-free-guidance as how closely to the prompt your image generation adheres. Lower is more creative, and higher is more close to the prompt. You should play with this until you find a value you like.

**Seed.** The diffusion process can be made deterministic or nondeterministic. Use a negative seed for a random (nondeterministic) generation, or set the seed to a value to make it deterministic.

**Embeddings Path.** This is a path to a directory on the machine that contains Stable Diffusion embeddings.

**Lora Model Dir.** A path to a directory on the machine that contains Stable Diffusion LoRA models.

**Diffusion Model Path.** Most Stable Diffusion models come with three components: A unet, a vae, and a clip model. If your model only contains a unet, use this setting to load it. You probably should not use it in conjunction with the Model Path property.

**Vae Path and Clip L Path.** These properties are explained above, but they *should* be used with *either* a model path, or a diffusion model path.

**Taesd Path.** TAEsd is low-cost autoencoder for Stable Diffusion that can very quickly decode latent space. Really, there is not a reason *not* to use this in the context of a video game. Stable-Diffusion.cpp will still need a vae for encoding, but will use the taesd for decoding latent space.

**Controlnet Path.** The path to a controlnet for use with your model. Take a look at the various controlnets available for SD1.5 and SDXL, and try and come up with some cool ideas.

**Control Strength.** The control strength determines how greatly the controlnet affects the generation.

**Schedule.** Choose the Stable Diffusion scheduler. There are enums bound for each of the schedules under the StableDiffusion namespace, using all caps.

**Sample Method.** Choose the sampling method for use with stable diffusion. This is another parameter that needs to be played with, but as a general rule of thumb, euler_a is probably the best. Use LCM if you plan to use an LCM weights-enabled model or LoRA.

**N Threads.** The N Threads parameter chooses the number of threads to run. You can use the get_cpu_count() method in the StableDiffusion node to smartly select a number of threads that does not exceed your number of physical cores. Use a number less than 1 to select all available cores.

**Wtype.** Convert the weights to any of the specified types. In general, you should not touch this unless you need to quantize weights to target lower-end devices. When you quantize weights, LoRA breaks.

**Clip Skip.** Skip the last N layers of clip. Do not use this setting in conjunction with LCM weights, but it makes anime-style images look very good when set to 1 or 2.

**Style Ratio.** I'll be honest, I have no idea what this does. I think it might be a part of the (as of right now) unsupported PHOTOMAKER api that stable-diffusion.cpp supports. I kept it in here just in case.

**Strength.** The noising/denoising strength determines how much noise is added/removed from the image during diffusion. In general, you could just keep this at its default value.

**Diffusion Flash Attn.** Enable Flash Attention. You should probably enable this in most cases.

**Vae Tiling.** When using Stable Diffussion on a GPU, this breaks the Variational Audtoencoder into sliding tiles, which lowers memory consumption (margianally).

**Control Net CPU.** When using Stable Diffusion on a GPU, this runs the Controlnet on the CPU. Useful to target lower-end devices.

**Clip on CPU, Vae on CPU.** See above.

**Batch Count.** Number of images to generate each time generation occurs. You should probably just keep this at 1 unless you're targeting high-end devices.

**RNG Type.** CUDA_RNG does not require CUDA devices, but if you run into problems with the CUDA_RNG algorithm, try changing this to STD_DEFAULT_RNG.

### The StableDiffusion Node --- Methods

```cpp
bool preload_ctx();
```

Create a Stable Diffusion context. If you need to generate multiple images during one scene, do this.

```cpp
void free_ctx();
```

Free the Stable Diffusion context. The destructor will also do this, but you can call this if you no longer need the Stable Diffusion context during a scene and would like to reclaim the VRAM.

```cpp
bool txt2img(
	String prompt = "",
	String negative_prompt = "",
	ImageTexture control_image = ImageTexture()
);
```

Generate an image using the prompt, negative prompt, and control image. By default, calling the method with no arguments causes the node to use the prompt and negative prompt properties. If you specify your own prompt, then the properties are overriden. A control net must be specified if you wish to use a control image.

```cpp
ImageTexture get_result(int result = 0);
```

Get the generated image as an ImageTexture. You must have generated an image/some images for this to return anything other than a null ImageTexture. If youc all this method with no arguments, it will simply use the first image you ever generated.

