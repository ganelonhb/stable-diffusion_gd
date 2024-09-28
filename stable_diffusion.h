#ifndef GODOT_STABLE_DIFFUSION_H
#define GODOT_STABLE_DIFFUSION_H

#include "scene/main/node.h"

#include <string.h>
#include <time.h>
#include <random>
#include <string>
#include <vector>

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

#include "core/string/ustring.h"

typedef struct sd_ctx_t sd_ctx_t;

class StableDiffusion : public Node {
	GDCLASS(StableDiffusion, Node);

protected:
	static void _bind_methods();

public:

	StableDiffusion();
	~StableDiffusion();

private:
	char *m_model_path_c_str{nullptr};
	char *m_clip_l_path_c_str{nullptr};
	char *m_t5xxl_path_c_str{nullptr};
	char *m_diffusion_model_path_c_str{nullptr};
	char *m_vae_path_c_str{nullptr};
	char *m_taesd_path_c_str{nullptr};
	char *m_control_net_path_c_str{nullptr};
	char *m_lora_model_dir_c_str{nullptr};
	char *m_embed_dir_c_str{nullptr};
	char *m_stacked_id_embed_dir_c_str{nullptr};

	bool m_vae_decode_only{false};
	bool m_vae_tiling{false};
	bool m_free_params_immediately{false};
	int m_n_threads{-1};
	int m_wtype{0};
	int m_rng_type{CUDA_RNG};
	int m_s{0};
	bool m_keep_clip_on_cpu{false};
	bool m_keep_control_net_cpu{false};
	bool m_keep_vae_on_cpu{false};

	sd_ctx_t *m_sd_ctx{nullptr};
};

#endif
