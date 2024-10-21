#ifndef GODOT_STABLE_DIFFUSION_H
#define GODOT_STABLE_DIFFUSION_H

// Godot Includes
#include "scene/main/node.h"
#include "core/string/ustring.h"
#include "core/object/class_db.h"
#include "core/variant/variant.h"
#include "scene/resources/texture.h"
#include "core/object/ref_counted.h"
#include "core/io/image.h"
#include "core/templates/vector.h"
#include "scene/resources/image_texture.h"
#include "core/os/thread.h"
#include "core/os/mutex.h"
#include "core/os/semaphore.h"

// C/C++ Includes
#include <string.h>
#include <time.h>
#include <stddef.h>
#include <random>
#include <string>
#include <vector>
#include <sstream>
#include <limits>
#include <cstdint>
#include <unordered_map>

// Stable Diffusion Includes
// #include "preprocessing.hpp"

#include "flux.hpp"
#include "stable-diffusion.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include "stb_image_resize.h"

extern const char *rng_type_to_str[];

extern const char *sample_method_str[];

extern const char *shedule_str[];

extern const char* modes_str[];

class StableDiffusion : public Node {
	GDCLASS(StableDiffusion, Node);

	public:
	enum SDMode : int {
		TXT2IMG = 0,
		IMG2IMG = 1,
		IMG2VID = 2, // Broken as of now.
		CONVERT = 3,
		MODE_COUNT = 4, // Must always be last.
	};

	struct SDParams {
		int n_threads = -1;
		SDMode mode   = TXT2IMG;

		std::string model_path;
		std::string clip_l_path;
		std::string t5xxl_path;
		std::string diffusion_model_path;
		std::string vae_path;
		std::string taesd_path;
		std::string esrgan_path;
		std::string controlnet_path;
		std::string embeddings_path;
		std::string stacked_id_embeddings_path;
		std::string input_id_images_path;
		sd_type_t wtype = SD_TYPE_COUNT;
		std::string lora_model_dir;
		std::string output_path = "output.png";
		std::string input_path;
		std::string control_image_path;

		std::string prompt;
		std::string negative_prompt;
		float min_cfg     = 1.0f;
		float cfg_scale   = 7.0f;
		float guidance    = 3.5f;
		float style_ratio = 20.f;
		int clip_skip     = -1;  // <= 0 represents unspecified
		int width         = 512;
		int height        = 512;
		int batch_count   = 1;

		int video_frames         = 6;
		int motion_bucket_id     = 127;
		int fps                  = 6;
		float augmentation_level = 0.f;

		sample_method_t sample_method = EULER_A;
		schedule_t schedule           = DEFAULT;
		int sample_steps              = 20;
		float strength                = 0.75f;
		float control_strength        = 0.9f;
		rng_type_t rng_type           = CUDA_RNG;
		int64_t seed                  = 42;
		bool verbose                  = false;
		bool vae_tiling               = false;
		bool control_net_cpu          = false;
		bool normalize_input          = false;
		bool clip_on_cpu              = false;
		bool vae_on_cpu               = false;
		bool diffusion_flash_attn	  = false;
		bool canny_preprocess         = false;
		bool color                    = false;
		int upscale_repeats           = 1;
	};

protected:
	static void _bind_methods();

public:

	StableDiffusion();
	~StableDiffusion();

	// Debug
	void test_sd(const String str);

	// Functionality

	bool preload_ctx();
	// void threaded_preload_ctx();
	void free_ctx();

	void txt2img(
		String prompt = String(),
		String negative_prompt = String(),
		Ref<Image> control_image = Ref<Image>()
	); // meow

	Ref<ImageTexture> get_result(int result = 0) const;
	Ref<ImageTexture> get_last_result() const;

	int get_num_cpus() const;

	// Parameters
	int get_n_threads() const;
	void set_n_threads(int n_threads=-1);

	int get_wtype() const;
	void set_wtype(int wtype = SD_TYPE_COUNT);

	float get_cfg_scale() const;
	void set_cfg_scale(float f=7.0f);

	float get_style_ratio() const;
	void set_style_ratio(float style_ratio=20.0f);

	int get_clip_skip() const;
	void set_clip_skip(int clip_skip = -1);

	int get_width() const;
	void set_width(int width = 512);

	int get_height() const;
	void set_height(int height = 512);

	int get_batch_count() const;
	void set_batch_count(int batch_count = 1);

	int get_sample_method() const;
	void set_sample_method(int sample_method = EULER_A);

	int get_schedule() const;
	void set_schedule(int schedule = DEFAULT);

	int get_sample_steps() const;
	void set_sample_steps(int sample_steps = 20);

	float get_strength() const;
	void set_strength(float strength = 0.75f);

	float get_control_strength() const;
	void set_control_strength(float control_strength = 0.9f);

	int get_rng_type() const;
	void set_rng_type(int rng_type = CUDA_RNG);

	int64_t get_seed() const;
	void set_seed(int64_t seed = 42);

	bool get_vae_tiling() const;
	void set_vae_tiling(bool vae_tiling = false);

	bool get_control_net_cpu() const;
	void set_control_net_cpu(bool control_net_cpu = false);

	bool get_clip_on_cpu() const;
	void set_clip_on_cpu(bool clip_on_cpu = false);

	bool get_vae_on_cpu() const;
	void set_vae_on_cpu(bool vae_on_cpu = false);

	bool get_diffusion_flash_attn() const;
	void set_diffusion_flash_attn(bool diffusion_flash_attn = false);

	String get_model_path() const;
	void set_model_path(String model_path);

	String get_clip_l_path() const;
	void set_clip_l_path(String clip_l_path);


	String get_diffusion_model_path() const;
	void set_diffusion_model_path(String diffusion_model_path);

	String get_vae_path() const;
	void set_vae_path(String vae_path);

	String get_taesd_path() const;
	void set_taesd_path(String taesd_path);

	String get_controlnet_path() const;
	void set_controlnet_path(String controlnet_path);

	String get_embeddings_path() const;
	void set_embeddings_path(String embeddings_path);

	String get_lora_model_dir() const;
	void set_lora_model_dir(String lora_model_dir);

	String get_prompt() const;
	void set_prompt(String prompt);

	String get_negative_prompt() const;
	void set_negative_prompt(String negative_prompt);

private:
	Thread thread;
	Mutex mutex;
	Semaphore semaphore;

	String prompt;
	String negative_prompt;
	Ref<Image> control_image;

	static void _txt2img_callback(void *p_user);
	void _on_txt2img_complete();

	static const int USE_MODEL_WEIGHTS{static_cast<int>(SD_TYPE_COUNT)};

	bool preloaded;

	SDParams m_params;

	Vector<Ref<ImageTexture>> m_results;

	sd_ctx_t *m_sd_ctx{nullptr};
};

#endif
