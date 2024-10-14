#include "stable_diffusion.h"

#include <iostream>

const char *rng_type_to_str[] = {
	"std_default",
	"cuda",
};

const char *sample_method_str[] = {
	"euler_a",
    "euler",
    "heun",
    "dpm2",
    "dpm++2s_a",
    "dpm++2m",
    "dpm++2mv2",
    "ipndm",
    "ipndm_v",
    "lcm",
};

const char *shedule_str[] = {
	"default",
    "discrete",
    "karras",
    "exponential",
    "ays",
    "gits",
};

const char* modes_str[] = {
    "txt2img",
    "img2img",
    "img2vid",
    "convert",
};


StableDiffusion::StableDiffusion()
	: preloaded(false)
{

}

StableDiffusion::~StableDiffusion() {
	if (m_sd_ctx)
		free_sd_ctx(m_sd_ctx);
}

bool StableDiffusion::preload_ctx() {
	if (preloaded || m_sd_ctx)
		free_sd_ctx(m_sd_ctx);

	m_sd_ctx = new_sd_ctx(
		m_params.model_path.c_str(),
		m_params.clip_l_path.c_str(),
		"",
		m_params.diffusion_model_path.c_str(),
		m_params.vae_path.c_str(),
		m_params.taesd_path.c_str(),
		m_params.controlnet_path.c_str(),
		m_params.lora_model_dir.c_str(),
		"",
		"",
		false,
		m_params.vae_tiling,
		true,
		m_params.n_threads,
		m_params.wtype,
		m_params.rng_type,
		m_params.schedule,
		m_params.clip_on_cpu,
		m_params.control_net_cpu,
		m_params.vae_on_cpu,
		m_params.diffusion_flash_attn
	);

	preloaded = m_sd_ctx != NULL;

	return preloaded;
}

bool StableDiffusion::txt2img(
	String prompt,
	String negative_prompt,
	Ref<ImageTexture> control_image
) {

	// load ctx
	if (!preloaded) {
		m_sd_ctx = new_sd_ctx(
			m_params.model_path.c_str(),
			m_params.clip_l_path.c_str(),
			"",
			m_params.diffusion_model_path.c_str(),
			m_params.vae_path.c_str(),
			m_params.taesd_path.c_str(),
			m_params.controlnet_path.c_str(),
			m_params.lora_model_dir.c_str(),
			"",
			"",
			false,
			m_params.vae_tiling,
			true,
			m_params.n_threads,
			m_params.wtype,
			m_params.rng_type,
			m_params.schedule,
			m_params.clip_on_cpu,
			m_params.control_net_cpu,
			m_params.vae_on_cpu,
			m_params.diffusion_flash_attn
		);
	}

	if (!m_sd_ctx) return false;

	// txt2img
	uint8_t *control_image_buffer = NULL;
	sd_image_t *sd_control_image = NULL;

	sd_image_t *results = NULL;

	Ref<Image> img = control_image != Ref<ImageTexture>() ? control_image->get_image() : Ref<Image>();

	if (m_params.controlnet_path.size() > 0 && !img.is_null()) {
		int width = img->get_width();
		int height = img->get_height();
		int channels = img->get_format() == Image::FORMAT_RGBA8 ? 4 : 3;
		const uint8_t *raw_data = img->get_data().ptr();

		int desired_channels = 0;
		control_image_buffer = stbi_load_from_memory(raw_data, width * height * channels, &width, &height, &desired_channels, desired_channels);

		if (control_image_buffer)
			print_line("-- Using a Controlnet --");

		sd_control_image = new sd_image_t{
			(uint32_t)m_params.width,
			(uint32_t)m_params.height,
			3,
			control_image_buffer};
	}

	std::string std_prompt = prompt != String() ? std::string(prompt.utf8()) : m_params.prompt;
	std::string std_nprompt = negative_prompt != String() ? std::string(negative_prompt.utf8()) : m_params.negative_prompt;

	results = ::txt2img(
		m_sd_ctx,
		std_prompt.c_str(),
		std_nprompt.c_str(),
		m_params.clip_skip,
		m_params.cfg_scale,
		m_params.guidance,
		m_params.width,
		m_params.height,
		m_params.sample_method,
		m_params.sample_steps,
		m_params.seed,
		m_params.batch_count,
		sd_control_image,
		m_params.control_strength,
		m_params.style_ratio,
		false,
		""
	);


	if (!results) return false;

	for (int i = 0; i < m_params.batch_count; ++i)
	{
		if (!results[i].data) continue;

		Ref<Image> image;

		int img_size = results[i].width * results[i].height * results[i].channel;
		Vector<uint8_t> imgdata;
		imgdata.resize(img_size);
		memcpy(imgdata.ptrw(), results[i].data, img_size);

		std::cout << imgdata.size() << std::endl;

		Image::Format format = results[i].channel == 4 ? Image::FORMAT_RGBA8 : Image::FORMAT_RGB8;
		image = Image::create_from_data(
			results[i].width,
			results[i].height,
			false,
			format,
			imgdata
		);

		Ref<ImageTexture> texture = ImageTexture::create_from_image(image);

		m_results.append(texture);

		free(results[i].data);
	}

	free(results);

	// free ctx
	if (!preloaded) {
		free_sd_ctx(m_sd_ctx);
		m_sd_ctx = nullptr;
	}

	if (sd_control_image) delete sd_control_image;
	if (control_image_buffer) free(control_image_buffer);
	return true;
}

Ref<ImageTexture> StableDiffusion::get_result(int result) const {
	if (!m_results.size()) return Ref<ImageTexture>();

	return m_results[result];
}

int StableDiffusion::get_num_cpus() const {
	return get_num_physical_cores();
}

int StableDiffusion::get_n_threads() const {
	return m_params.n_threads;
}

void StableDiffusion::set_n_threads(int n_threads) {
	m_params.n_threads = n_threads;
}

int StableDiffusion::get_wtype() const {
	return m_params.wtype;
}

void StableDiffusion::set_wtype(int wtype) {
	if (wtype == 4 || wtype == 5) return;

	m_params.wtype = static_cast<sd_type_t>(wtype);
}

float StableDiffusion::get_cfg_scale() const {
	return m_params.cfg_scale;
}

void StableDiffusion::set_cfg_scale(float cfg_scale) {
	m_params.cfg_scale = cfg_scale;
}

float StableDiffusion::get_style_ratio() const {
	return m_params.style_ratio;
}

void StableDiffusion::set_style_ratio(float style_ratio) {
	m_params.style_ratio = style_ratio;
}

int StableDiffusion::get_clip_skip() const {
	return m_params.clip_skip;
}

void StableDiffusion::set_clip_skip(int clip_skip) {
	m_params.clip_skip = clip_skip;
}

int StableDiffusion::get_width() const {
	return  m_params.width;
}

void StableDiffusion::set_width(int width) {
	m_params.width = width;
}

int StableDiffusion::get_height() const {
	return m_params.height;
}

void StableDiffusion::set_height(int height) {
	m_params.height = height;
}

int StableDiffusion::get_batch_count() const {
	return m_params.batch_count;
}

void StableDiffusion::set_batch_count(int batch_count) {
	m_params.batch_count = batch_count ? std::abs(batch_count) : 1;
}

int StableDiffusion::get_sample_method() const {
	return m_params.sample_method;
}

void StableDiffusion::set_sample_method(int sample_method) {
	m_params.sample_method = static_cast<sample_method_t>(sample_method);
}

int StableDiffusion::get_schedule() const {
	return m_params.schedule;
}

void StableDiffusion::set_schedule(int schedule) {
	m_params.schedule = static_cast<schedule_t>(schedule);
}

int StableDiffusion::get_sample_steps() const {
	return m_params.sample_steps;
}

void StableDiffusion::set_sample_steps(int sample_steps) {
	m_params.sample_steps = sample_steps ? std::abs(sample_steps) : 1;
}

float StableDiffusion::get_strength() const {
	return m_params.strength;
}

void StableDiffusion::set_strength(float strength) {
	m_params.strength = strength;
}

float StableDiffusion::get_control_strength() const {
	return m_params.control_strength;
}

void StableDiffusion::set_control_strength(float control_strength) {
	m_params.control_strength = control_strength;
}

int StableDiffusion::get_rng_type() const {
	return m_params.rng_type;
}

void StableDiffusion::set_rng_type(int rng_type) {
	m_params.rng_type = static_cast<rng_type_t>(rng_type);
}

int64_t StableDiffusion::get_seed() const {
	return m_params.seed;
}

void StableDiffusion::set_seed(int64_t seed) {
	m_params.seed = seed;
}

bool StableDiffusion::get_vae_tiling() const {
	return m_params.vae_tiling;
}

void StableDiffusion::set_vae_tiling(bool vae_tiling) {
	m_params.vae_tiling = vae_tiling;
}

bool StableDiffusion::get_control_net_cpu() const {
	return m_params.control_net_cpu;
}

void StableDiffusion::set_control_net_cpu(bool control_net_cpu) {
	m_params.control_net_cpu = control_net_cpu;
}

bool StableDiffusion::get_clip_on_cpu() const {
	return m_params.clip_on_cpu;
}

void StableDiffusion::set_clip_on_cpu(bool clip_on_cpu) {
	m_params.clip_on_cpu = clip_on_cpu;
}

bool StableDiffusion::get_vae_on_cpu() const {
	return m_params.vae_on_cpu;
}

void StableDiffusion::set_vae_on_cpu(bool vae_on_cpu) {
	m_params.vae_on_cpu = vae_on_cpu;
}

bool StableDiffusion::get_diffusion_flash_attn() const {
	return m_params.diffusion_flash_attn;
}

void StableDiffusion::set_diffusion_flash_attn(bool diffusion_flash_attn) {
	m_params.diffusion_flash_attn = diffusion_flash_attn;
}

String StableDiffusion::get_model_path() const {
	return String::utf8(m_params.model_path.c_str());
}

void StableDiffusion::set_model_path(String model_path) {
	m_params.model_path = model_path.utf8();
}

String StableDiffusion::get_clip_l_path() const {
	return String::utf8(m_params.clip_l_path.c_str());
}

void StableDiffusion::set_clip_l_path(String clip_l_path) {
	m_params.clip_l_path = clip_l_path.utf8();
}

String StableDiffusion::get_diffusion_model_path() const {
	return String::utf8(m_params.diffusion_model_path.c_str());
}

void StableDiffusion::set_diffusion_model_path(String diffusion_model_path) {
	m_params.diffusion_model_path = diffusion_model_path.utf8();
}

String StableDiffusion::get_vae_path() const {
	return String::utf8(m_params.vae_path.c_str());
}

void StableDiffusion::set_vae_path(String vae_path) {
	m_params.vae_path = vae_path.utf8();
}

String StableDiffusion::get_taesd_path() const {
	return String::utf8(m_params.taesd_path.c_str());
}

void StableDiffusion::set_taesd_path(String taesd_path) {
	m_params.taesd_path = taesd_path.utf8();
}

String StableDiffusion::get_controlnet_path() const {
	return String::utf8(m_params.controlnet_path.c_str());
}

void StableDiffusion::set_controlnet_path(String controlnet_path) {
	m_params.controlnet_path = controlnet_path.utf8();
}

String StableDiffusion::get_embeddings_path() const {
	return String::utf8(m_params.embeddings_path.c_str());
}

void StableDiffusion::set_embeddings_path(String embeddings_path) {
	m_params.embeddings_path = embeddings_path.utf8();
}

String StableDiffusion::get_lora_model_dir() const {
	return String::utf8(m_params.lora_model_dir.c_str());
}

void StableDiffusion::set_lora_model_dir(String lora_model_dir) {
	m_params.lora_model_dir = lora_model_dir.utf8();
}

String StableDiffusion::get_prompt() const {
	return String::utf8(m_params.prompt.c_str());
}

void StableDiffusion::set_prompt(String prompt) {
	m_params.prompt = prompt.utf8();
}

String StableDiffusion::get_negative_prompt() const {
	return String::utf8(m_params.negative_prompt.c_str());
}

void StableDiffusion::set_negative_prompt(String negative_prompt) {
	m_params.negative_prompt = negative_prompt.utf8();
}

void StableDiffusion::test_sd(const String str) {
	print_line("Started initializing SD context...");
	m_sd_ctx = new_sd_ctx(
		m_params.model_path.c_str(),
		m_params.clip_l_path.c_str(),
		m_params.t5xxl_path.c_str(),
		m_params.diffusion_model_path.c_str(),
		m_params.vae_path.c_str(),
		m_params.taesd_path.c_str(),
		m_params.controlnet_path.c_str(),
		"/home/donquixote/git/models/lora",
		"",
		"",
		false,
		m_params.vae_tiling,
		true,
		m_params.n_threads,
		m_params.wtype,
		m_params.rng_type,
		m_params.schedule,
		m_params.clip_on_cpu,
		m_params.control_net_cpu,
		m_params.vae_on_cpu,
		m_params.diffusion_flash_attn
	);

	if (m_sd_ctx == NULL) {
		print_line("SD Context failed");
	}
	else {
		print_line("SD Context created!");
	}

	sd_image_t * results;

	results = ::txt2img(
		m_sd_ctx,
		str.utf8().get_data(),
		m_params.negative_prompt.c_str(),
		m_params.clip_skip,
		m_params.cfg_scale,
		m_params.guidance,
		m_params.width,
		m_params.height,
		m_params.sample_method,
		m_params.sample_steps,
		m_params.seed,
		m_params.batch_count,
		NULL,
		m_params.control_strength,
		m_params.style_ratio,
		m_params.normalize_input,
		""
	);

	if (results == NULL) {
		print_line("Generation failed!");
		return;
	}

	stbi_write_png(
		"/home/donquixote/git/godot/output.png",
		results[0].width,
		results[0].height,
		results[0].channel,
		results[0].data,
		0,
		""
	);

	free(results[0].data);
	free(results);
}

void StableDiffusion::_bind_methods() {
	ClassDB::bind_method(D_METHOD("test_sd"), &StableDiffusion::test_sd);

	std::stringstream quants;
	quants	<< "f32," << "f16," << "q4_0," << "q4_1,"
			<< "DEPRECATED q4_2," << "DEPRECATED q4_3,"
			<< "q5_0," << "q5_1," << "q8_0," << "q8_1,"
			<< "q2_k," << "q3_k," << "q4_k," << "q5_k,"
			<< "q6_k," << "q8_k," << "iq2_xxs," << "iq2_xs,"
			<< "iq3_xxs," << "iq1_s," << "iq4_nl," << "iq3_s,"
			<< "iq2_s," << "iq4_xs," << "i8," << "i16,"
			<< "i32," << "i64," << "f64," << "iq1_m,"
			<< "bf16," << "q4_0_4_4," << "q4_0_4_8,"
			<< "q4_0_8_8," << "model weights";

	// Class Methods

	ClassDB::bind_method(
		D_METHOD("txt2img", "prompt", "negative_prompt", "control_image"),
		&StableDiffusion::txt2img,
		DEFVAL(String()),
		DEFVAL(String()),
		DEFVAL(Ref<ImageTexture>())
	);

	ClassDB::bind_method(D_METHOD("get_result", "result"), &StableDiffusion::get_result, DEFVAL(0));

	ClassDB::bind_method(D_METHOD("get_num_cpus"), &StableDiffusion::get_num_cpus);

	ClassDB::bind_method(D_METHOD("set_n_threads", "n_threads"), &StableDiffusion::set_n_threads, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("get_n_threads"), &StableDiffusion::get_n_threads);

	ClassDB::bind_method(D_METHOD("set_wtype", "wtype"), &StableDiffusion::set_wtype, DEFVAL(static_cast<int>(SD_TYPE_COUNT)));
	ClassDB::bind_method(D_METHOD("get_wtype"), &StableDiffusion::get_wtype);

	ClassDB::bind_method(D_METHOD("set_cfg_scale", "cfg_scale"), &StableDiffusion::set_cfg_scale, DEFVAL(7.0f));
	ClassDB::bind_method(D_METHOD("get_cfg_scale"), &StableDiffusion::get_cfg_scale);

	ClassDB::bind_method(D_METHOD("set_style_ratio", "style_ratio"), &StableDiffusion::set_style_ratio, DEFVAL(20.0f));
	ClassDB::bind_method(D_METHOD("get_style_ratio"), &StableDiffusion::get_style_ratio);

	ClassDB::bind_method(D_METHOD("set_clip_skip", "clip_skip"), &StableDiffusion::set_clip_skip, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("get_clip_skip"), &StableDiffusion::get_clip_skip);

	ClassDB::bind_method(D_METHOD("set_width", "width"), &StableDiffusion::set_width, DEFVAL(512));
	ClassDB::bind_method(D_METHOD("get_width"), &StableDiffusion::get_width);

	ClassDB::bind_method(D_METHOD("set_height", "height"), &StableDiffusion::set_height, DEFVAL(512));
	ClassDB::bind_method(D_METHOD("get_height"), &StableDiffusion::get_height);

	ClassDB::bind_method(D_METHOD("set_batch_count", "batch_count"), &StableDiffusion::set_batch_count, DEFVAL(1));
	ClassDB::bind_method(D_METHOD("get_batch_count"), &StableDiffusion::get_batch_count);

	ClassDB::bind_method(D_METHOD("set_sample_method", "sample_method"), &StableDiffusion::set_sample_method, DEFVAL(EULER_A));
	ClassDB::bind_method(D_METHOD("get_sample_method"), &StableDiffusion::get_sample_method);

	ClassDB::bind_method(D_METHOD("set_schedule", "schedule"), &StableDiffusion::set_schedule, DEFVAL(DEFAULT));
	ClassDB::bind_method(D_METHOD("get_schedule"), &StableDiffusion::get_schedule);

	ClassDB::bind_method(D_METHOD("set_sample_steps", "sample_steps"), &StableDiffusion::set_sample_steps, DEFVAL(20));
	ClassDB::bind_method(D_METHOD("get_sample_steps"), &StableDiffusion::get_sample_steps);

	ClassDB::bind_method(D_METHOD("set_strength", "strength"), &StableDiffusion::set_strength, DEFVAL(0.75f));
	ClassDB::bind_method(D_METHOD("get_strength"), &StableDiffusion::get_strength);

	ClassDB::bind_method(D_METHOD("set_control_strength", "control_strength"), &StableDiffusion::set_control_strength, DEFVAL(0.9f));
	ClassDB::bind_method(D_METHOD("get_control_strength"), &StableDiffusion::get_control_strength);

	ClassDB::bind_method(D_METHOD("set_rng_type", "rng_type"), &StableDiffusion::set_rng_type, DEFVAL(CUDA_RNG));
	ClassDB::bind_method(D_METHOD("get_rng_type"), &StableDiffusion::get_rng_type);

	ClassDB::bind_method(D_METHOD("set_seed", "seed"), &StableDiffusion::set_seed, DEFVAL(static_cast<int64_t>(42)));
	ClassDB::bind_method(D_METHOD("get_seed"), &StableDiffusion::get_seed);

	ClassDB::bind_method(D_METHOD("set_vae_tiling", "vae_tiling"), &StableDiffusion::set_vae_tiling, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_vae_tiling"), &StableDiffusion::get_vae_tiling);

	ClassDB::bind_method(D_METHOD("set_control_net_cpu", "control_net_cpu"), &StableDiffusion::set_control_net_cpu, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_control_net_cpu"), &StableDiffusion::get_control_net_cpu);

	ClassDB::bind_method(D_METHOD("set_clip_on_cpu", "clip_on_cpu"), &StableDiffusion::set_clip_on_cpu, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_clip_on_cpu"), &StableDiffusion::get_clip_on_cpu);

	ClassDB::bind_method(D_METHOD("set_vae_on_cpu", "vae_on_cpu"), &StableDiffusion::set_vae_on_cpu, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_vae_on_cpu"), &StableDiffusion::get_vae_on_cpu);

	ClassDB::bind_method(D_METHOD("set_diffusion_flash_attn", "diffusion_flash_attn"), &StableDiffusion::set_diffusion_flash_attn, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_diffusion_flash_attn"), &StableDiffusion::get_diffusion_flash_attn);

	ClassDB::bind_method(D_METHOD("set_model_path", "model_path"), &StableDiffusion::set_model_path);
	ClassDB::bind_method(D_METHOD("get_model_path"), &StableDiffusion::get_model_path);

	ClassDB::bind_method(D_METHOD("set_clip_l_path", "clip_l_path"), &StableDiffusion::set_clip_l_path);
	ClassDB::bind_method(D_METHOD("get_clip_l_path"), &StableDiffusion::get_clip_l_path);

	ClassDB::bind_method(D_METHOD("set_diffusion_model_path", "diffusion_model_path"), &StableDiffusion::set_diffusion_model_path);
	ClassDB::bind_method(D_METHOD("get_diffusion_model_path"), &StableDiffusion::get_diffusion_model_path);

	ClassDB::bind_method(D_METHOD("set_vae_path", "vae_path"), &StableDiffusion::set_vae_path);
	ClassDB::bind_method(D_METHOD("get_vae_path"), &StableDiffusion::get_vae_path);

	ClassDB::bind_method(D_METHOD("set_taesd_path", "taesd_path"), &StableDiffusion::set_taesd_path);
	ClassDB::bind_method(D_METHOD("get_taesd_path"), &StableDiffusion::get_taesd_path);

	ClassDB::bind_method(D_METHOD("set_controlnet_path", "controlnet_path"), &StableDiffusion::set_controlnet_path);
	ClassDB::bind_method(D_METHOD("get_controlnet_path"), &StableDiffusion::get_controlnet_path);

	ClassDB::bind_method(D_METHOD("set_embeddings_path", "embeddings_path"), &StableDiffusion::set_embeddings_path);
	ClassDB::bind_method(D_METHOD("get_embeddings_path"), &StableDiffusion::get_embeddings_path);

	ClassDB::bind_method(D_METHOD("set_lora_model_dir", "lora_model_dir"), &StableDiffusion::set_lora_model_dir);
	ClassDB::bind_method(D_METHOD("get_lora_model_dir"), &StableDiffusion::get_lora_model_dir);

	ClassDB::bind_method(D_METHOD("set_prompt", "prompt"), &StableDiffusion::set_prompt);
	ClassDB::bind_method(D_METHOD("get_prompt"), &StableDiffusion::get_prompt);

	ClassDB::bind_method(D_METHOD("set_negative_prompt", "negative_prompt"), &StableDiffusion::set_negative_prompt);
	ClassDB::bind_method(D_METHOD("get_negative_prompt"), &StableDiffusion::get_negative_prompt);


	// Properties
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "prompt"), "set_prompt", "get_prompt");

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "negative_prompt"), "set_negative_prompt", "get_negative_prompt");

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "model_path", PROPERTY_HINT_GLOBAL_FILE), "set_model_path", "get_model_path");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "width"), "set_width", "get_width");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "height"), "set_height", "get_height");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "sample_steps"), "set_sample_steps", "get_sample_steps");

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cfg_scale", PROPERTY_HINT_RANGE, "0.0,9999.0,0.1", PROPERTY_USAGE_DEFAULT, "The Classifier-Free Guidance scale determines how closely the model adheres to the prompt. Lower value is more creative, wheras higher value is closer to the prompt."), "set_cfg_scale", "get_cfg_scale");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "seed"), "set_seed", "get_seed");

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "embeddings_path", PROPERTY_HINT_GLOBAL_FILE), "set_embeddings_path", "get_embeddings_path");

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "lora_model_dir", PROPERTY_HINT_GLOBAL_DIR), "set_lora_model_dir", "get_lora_model_dir");

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "diffusion_model_path", PROPERTY_HINT_GLOBAL_FILE), "set_diffusion_model_path", "get_diffusion_model_path");

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "vae_path", PROPERTY_HINT_GLOBAL_FILE), "set_vae_path", "get_vae_path");

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "taesd_path", PROPERTY_HINT_GLOBAL_FILE), "set_taesd_path", "get_taesd_path");

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "clip_l_path", PROPERTY_HINT_GLOBAL_FILE), "set_clip_l_path", "get_clip_l_path");

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "controlnet_path", PROPERTY_HINT_GLOBAL_FILE), "set_controlnet_path", "get_controlnet_path");

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "control_strength", PROPERTY_HINT_RANGE, "0.0,5.0,0.01"), "set_control_strength", "get_control_strength");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "schedule", PROPERTY_HINT_ENUM, "Default,Discrete,Karras,Exponential,AYS,GITS"), "set_schedule", "get_schedule");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "sample_method", PROPERTY_HINT_ENUM, "euler_a,euler,heun,dpm2,dpm++2s_a,dpm++2m,dpm++2mv2,ipndm,ipndm_v,lcm", PROPERTY_USAGE_DEFAULT, "The sampling method dictates how the model diffuses the image. Different options have different results on the quality of the image, depending on what you are generating. euler_a is reccomended for most purposes."), "set_sample_method", "get_sample_method");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "n_threads", PROPERTY_HINT_RANGE, "-1,99,1", PROPERTY_USAGE_DEFAULT, "How many threads to use with Stable Diffusion. n_threads <= 0 will use all available physical cores."), "set_n_threads", "get_n_threads");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "wtype", PROPERTY_HINT_ENUM, quants.str().c_str(), PROPERTY_USAGE_DEFAULT, "Weight types to use. Quantizing weights causes LORA to look bad. It is reccomended to only use options provided by the command line version of stable-diffusion.cpp."), "set_wtype", "get_wtype");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "clip_skip", PROPERTY_HINT_RANGE, "-1,99,1", PROPERTY_USAGE_DEFAULT, "Clip skip can be used to modify how the prompt is interpreted, allowing for more creative or even abstract interpretations of the prompt."), "set_clip_skip", "get_clip_skip");

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "style_ratio", PROPERTY_HINT_RANGE, "0.0,100.0,0.1"), "set_style_ratio", "get_style_ratio");

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "strength", PROPERTY_HINT_RANGE, "0.0,5.0,0.01"), "set_strength", "get_strength");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "diffusion_flash_attn"), "set_diffusion_flash_attn", "get_diffusion_flash_attn");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "vae_tiling"), "set_vae_tiling", "get_vae_tiling");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "control_net_cpu"), "set_control_net_cpu", "get_control_net_cpu");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "clip_on_cpu"), "set_clip_on_cpu", "get_clip_on_cpu");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "vae_on_cpu"), "set_vae_on_cpu", "get_vae_on_cpu");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "batch_count"), "set_batch_count", "get_batch_count");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "rng_type", PROPERTY_HINT_ENUM, "STD Default RNG, CUDA RNG"), "set_rng_type", "get_rng_type");

	// Mode Enum
	BIND_CONSTANT(TXT2IMG);
	BIND_CONSTANT(IMG2IMG);
	BIND_CONSTANT(CONVERT);

	// sdtype_t enum
	BIND_CONSTANT(SD_TYPE_F32);
	BIND_CONSTANT(SD_TYPE_F16);
	BIND_CONSTANT(SD_TYPE_Q4_0);
	BIND_CONSTANT(SD_TYPE_Q4_1);
	BIND_CONSTANT(SD_TYPE_Q5_0);
	BIND_CONSTANT(SD_TYPE_Q5_1);
	BIND_CONSTANT(SD_TYPE_Q8_0);
	BIND_CONSTANT(SD_TYPE_Q2_K);
	BIND_CONSTANT(SD_TYPE_Q3_K);
	BIND_CONSTANT(SD_TYPE_Q4_K);
	BIND_CONSTANT(USE_MODEL_WEIGHTS);


	// sample_method_t enum
	BIND_CONSTANT(EULER_A);
	BIND_CONSTANT(EULER);
	BIND_CONSTANT(HEUN);
	BIND_CONSTANT(DPM2);
	BIND_CONSTANT(DPMPP2S_A);
	BIND_CONSTANT(DPMPP2M);
	BIND_CONSTANT(DPMPP2Mv2);
	BIND_CONSTANT(IPNDM);
	BIND_CONSTANT(IPNDM_V);
	BIND_CONSTANT(LCM);

	// schedule_t enum
	BIND_CONSTANT(DEFAULT);
	BIND_CONSTANT(DISCRETE);
	BIND_CONSTANT(KARRAS);
	BIND_CONSTANT(EXPONENTIAL);
	BIND_CONSTANT(AYS);
	BIND_CONSTANT(GITS);

	// rng_type_t enum
	BIND_CONSTANT(STD_DEFAULT_RNG);
	BIND_CONSTANT(CUDA_RNG);
}
