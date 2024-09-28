#include "stable_diffusion.h"

StableDiffusion::StableDiffusion() {}
StableDiffusion::~StableDiffusion() {
	if (sd)
		free_sd_ctx(sd);

	if (model_path_c_str)
		delete model_path_c_str;

	if (clip_l_path_c_str)
		delete clip_l_path_c_str;

	if (t5xxl_path_c_str)
		delete t5xxl_path_c_str;

	if (diffusion_model_path_c_str)
		delete diffusion_model_path_c_str;

	if (vae_path_c_str)
		delete vae_path_c_str;

	if (taesd_path_c_str)
		delete taesd_path_c_str;

	if (control_net_path_c_str)
		delete control_net_path_c_str;

	if (lora_model_dir_c_str)
		delete lora_model_dir_c_str;

	if (embed_dir_c_str)
		delete embed_dir_c_str;

	if (stacked_id_embed_dir_c_str)
		delete stacked_id_embed_dir_c_str;
}

void StableDiffusion::_bind_methods() {

}

bool StableDiffusion::godot_sd_ctx(
	String model_path,
	String clip_l_path,
	String t5xxl_path,
	String diffusion_model_path,
	String vae_path,
	String taesd_path,
	String control_net_path,
	String lora_model_dir,
	String embed_dir,
	String stacked_id_embed_dir,
	bool vae_decode_only,
	bool vae_tiling,
	bool free_params_immediately,
	int n_threads,
	int wtype,
	int rng_type,
	int s,
	bool keep_clip_on_cpu,
	bool keep_control_net_cpu,
	bool keep_vae_on_cpu
	) {



		return true;
	}
