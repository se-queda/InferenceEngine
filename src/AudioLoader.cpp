#include "audioguard/AudioLoader.h"
#include <iostream>

// Standard includes (Clean)
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswresample/swresample.h>
#include <libavutil/opt.h>
#include <libavutil/channel_layout.h>
}

namespace audioguard {

struct FFMpegResources {
    AVFormatContext* fmt_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* packet = nullptr;
    SwrContext* swr_ctx = nullptr;

    ~FFMpegResources() {
        if (frame) av_frame_free(&frame);
        if (packet) av_packet_free(&packet);
        if (codec_ctx) avcodec_free_context(&codec_ctx);
        if (fmt_ctx) avformat_close_input(&fmt_ctx);
        if (swr_ctx) swr_free(&swr_ctx);
    }
};

std::vector<float> AudioLoader::load_audio(const std::string& filepath) {
    FFMpegResources res;
    char err_buf[256];

    // 1. Open File
    int ret = avformat_open_input(&res.fmt_ctx, filepath.c_str(), nullptr, nullptr);
    if (ret < 0) {
        av_strerror(ret, err_buf, sizeof(err_buf));
        throw std::runtime_error("Could not open audio file: " + filepath + " (" + err_buf + ")");
    }

    if (avformat_find_stream_info(res.fmt_ctx, nullptr) < 0) {
        throw std::runtime_error("Could not find stream info.");
    }

    // 2. Find Audio Stream
    const AVCodec* codec = nullptr;
    int stream_idx = av_find_best_stream(res.fmt_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, &codec, 0);
    if (stream_idx < 0) {
        throw std::runtime_error("No audio stream found.");
    }

    // 3. Init Decoder
    res.codec_ctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(res.codec_ctx, res.fmt_ctx->streams[stream_idx]->codecpar);
    
    if (avcodec_open2(res.codec_ctx, codec, nullptr) < 0) {
        throw std::runtime_error("Failed to open codec.");
    }

    // 4. Init Resampler
    res.swr_ctx = swr_alloc();
    
    AVChannelLayout out_layout = AV_CHANNEL_LAYOUT_MONO;
    AVChannelLayout in_layout = res.codec_ctx->ch_layout;
    
    // Layout detection fallback
    if (in_layout.nb_channels <= 0) {
         av_channel_layout_copy(&in_layout, &res.fmt_ctx->streams[stream_idx]->codecpar->ch_layout);
    }
    
    // Guess layout from channel count (using nb_channels, NOT deprecated 'channels')
    if (in_layout.nb_channels <= 0) {
        // FIX: Access channel count via struct, not integer
        int channels = res.codec_ctx->ch_layout.nb_channels; 
        
        if (channels == 1) in_layout = AV_CHANNEL_LAYOUT_MONO;
        else if (channels == 2) in_layout = AV_CHANNEL_LAYOUT_STEREO;
        else throw std::runtime_error("Could not detect input audio channel layout.");
    }

    // Configure Resampler
    ret = swr_alloc_set_opts2(&res.swr_ctx,
                              &out_layout, AV_SAMPLE_FMT_FLT, 16000,
                              &in_layout, res.codec_ctx->sample_fmt, res.codec_ctx->sample_rate,
                              0, nullptr);
    
    if (ret < 0 || swr_init(res.swr_ctx) < 0) {
        throw std::runtime_error("Failed to initialize resampling context.");
    }

    // 5. Decode Loop
    res.frame = av_frame_alloc();
    res.packet = av_packet_alloc();
    std::vector<float> audio_buffer;

    while (av_read_frame(res.fmt_ctx, res.packet) >= 0) {
        if (res.packet->stream_index == stream_idx) {
            if (avcodec_send_packet(res.codec_ctx, res.packet) == 0) {
                while (avcodec_receive_frame(res.codec_ctx, res.frame) == 0) {
                    
                    int max_out = av_rescale_rnd(
                        swr_get_delay(res.swr_ctx, res.codec_ctx->sample_rate) + res.frame->nb_samples,
                        16000, res.codec_ctx->sample_rate, AV_ROUND_UP
                    );

                    if (max_out > 0) {
                        std::vector<float> data(max_out);
                        uint8_t* out_ptrs[1] = { (uint8_t*)data.data() };

                        int samples = swr_convert(res.swr_ctx, 
                                                  out_ptrs, max_out, 
                                                  (const uint8_t**)res.frame->data, res.frame->nb_samples);

                        if (samples > 0) {
                            audio_buffer.insert(audio_buffer.end(), data.begin(), data.begin() + samples);
                        }
                    }
                }
            }
        }
        av_packet_unref(res.packet);
    }
    
    // Flush
    int max_out = av_rescale_rnd(swr_get_delay(res.swr_ctx, res.codec_ctx->sample_rate), 16000, res.codec_ctx->sample_rate, AV_ROUND_UP);
    if (max_out > 0) {
        std::vector<float> data(max_out);
        uint8_t* out_ptrs[1] = { (uint8_t*)data.data() };
        int samples = swr_convert(res.swr_ctx, out_ptrs, max_out, nullptr, 0);
        if (samples > 0) {
             audio_buffer.insert(audio_buffer.end(), data.begin(), data.begin() + samples);
        }
    }

    return audio_buffer;
}

} // namespace audioguard