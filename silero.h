#pragma once

#if !defined(VADC_SLOW)
#define VADC_SLOW 0
#endif // VADC_SLOW

#include "tensor.h"

#include "conv.c"
#include "misc.c"
#include "stft.c"
#include "lstm.c"
#include "transformer.c"
#include "silero_v3.c"

#define MATHS_IMPLEMENTATION
#include "maths.h"

#include "silero_v31_16k_weights.c"

static void *silero_init(MemoryArena *arena, String8 model_path_arg, Silero_Config *config)
{
   VAR_UNUSED( model_path_arg );

   Silero_Context *silero_context = pushStruct(arena, Silero_Context);
   LoadTesttensorResult silero_weights_res = {0};

   silero_weights_res = load_testtensor_from_bytes(arena, sizeof(silero_v31_16k_weights), silero_v31_16k_weights );

   Assert ( silero_weights_res.tensor_count > 0 );
   int encoder_weights_count = 24 + 24 + 22 + 24;

   Assert( silero_weights_res.tensor_count == (1 + encoder_weights_count + 2 + 2) );

   silero_context->weights = silero_weights_init( silero_weights_res );
   silero_context->state_lstm_h = tensor_zeros_3d(arena, 2, 1, 64);
   silero_context->state_lstm_c = tensor_zeros_3d(arena, 2, 1, 64);

   config->batch_size_restriction = 1;
   config->is_silero_v5 = false;
   config->input_size_min = 1536;
   config->input_size_max = 1536;
   config->output_dims = 3;

   return silero_context;
}

static inline void *backend_init( MemoryArena *arena, String8 model_path_arg, Silero_Config *config)
{
   return silero_init(arena, model_path_arg, config);
}

static inline void backend_run(MemoryArena *arena, void *context_)
{
   VADC_Context *context = context_;

   // int output_stride = context->is_silero_v4 ? 1 : 2;

   // NOTE(irwin): hardcoded to v3.1 stride for now, v4 C version isn't implemented yet.
   int output_stride = 2;
   int silero_probability_out_index = 1;

   // TODO(irwin): dehardcode one batch
   One_Batch_Result result = silero_run_one_batch_with_context(arena,
                                                               context->backend,
                                                               context->buffers.window_size_samples,
                                                               context->buffers.input_samples);
   context->buffers.output[0 * output_stride + silero_probability_out_index] = result.prob;
}

static inline void backend_create_tensors(Silero_Config config, void *backend, Tensor_Buffers buffers)
{
   VAR_UNUSED(config);
   VAR_UNUSED(backend);
   VAR_UNUSED(buffers);
}
