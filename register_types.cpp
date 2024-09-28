#include "register_types.h"

#include "core/object/class_db.h"
#include "core/io/resource_loader.h"
#include "scene/resources/texture.h"
#include "stable_diffusion.h"

void initialize_stable_diffusion_gd_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	ClassDB::register_class<StableDiffusion>();
}

void uninitialize_stable_diffusion_gd_module(ModuleInitializationLevel p_level) {
}
