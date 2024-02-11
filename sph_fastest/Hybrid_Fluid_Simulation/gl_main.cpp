#include <sstream>
#include <cuda_runtime.h>
#include "sph_timer.h"
#include "sph_data.h"
#include "sph_hybrid_system.h"
#include "gl_main_header.h"

sph::HybridSystem* sph_system;

void init_sph_system()
{
    real_world_origin.x = -40.0f;
    real_world_origin.y = -40.0f;
    real_world_origin.z = -40.0f;

    real_world_side.x = 80.0f;
    real_world_side.y = 80.0f;
    real_world_side.z = 80.0f;

    sph_system = new sph::HybridSystem(real_world_side, real_world_origin);
}

bool init_cuda(void)
{
    int count = 0;
    int i = 0;

    cudaGetDeviceCount(&count);
    if (count == 0)
    {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    for (i = 0; i < count; i++)
    {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
        {
            if (prop.major >= 1)
            {
                break;
            }
        }
    }

    if (i == count)
    {
        fprintf(stderr, "There is no device supporting CUDA.\n");
        return false;
    }

    cudaSetDevice(i);

    printf("CUDA initialized.\n");
    return true;
}

int main(int argc, char** argv)
{

    if (!init_cuda()) return -1;

    init_sph_system();

    for (int i = 0; i < 100; i++) {
        printf("time step:  %d\n", i);
        sph_system->tick();
    }

    return 0;
}