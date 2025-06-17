#include "obj_s3_client.h"
#include <aws/core/utils/stream/PreallocatedStreamBuf.h>
#include <aws/core/utils/memory/stl/AWSStringStream.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <absl/strings/str_format.h>
#include "nixl_types.h"

namespace {
Aws::Client::ClientConfiguration
createClientConfiguration (nixl_b_params_t *custom_params) {
    Aws::Client::ClientConfiguration config;

    if (!custom_params) {
        return config;
    }

    auto endpoint_override_it = custom_params->find ("endpoint_override");
    if (endpoint_override_it != custom_params->end()) {
        config.endpointOverride = endpoint_override_it->second;
    }

    auto scheme_it = custom_params->find ("scheme");
    if (scheme_it != custom_params->end()) {
        if (scheme_it->second == "http") {
            config.scheme = Aws::Http::Scheme::HTTP;
        } else if (scheme_it->second == "https") {
            config.scheme = Aws::Http::Scheme::HTTPS;
        } else {
            throw std::runtime_error ("Invalid scheme: " + scheme_it->second);
        }
    }

    auto region_it = custom_params->find ("region");
    if (region_it != custom_params->end()) {
        config.region = region_it->second;
    }

    return config;
}

std::optional<Aws::Auth::AWSCredentials>
createAWSCredentials (nixl_b_params_t *custom_params) {
    if (!custom_params) {
        return std::nullopt;
    }

    std::string access_key, secret_key, session_token;

    auto access_key_it = custom_params->find ("access_key");
    if (access_key_it != custom_params->end()) {
        access_key = access_key_it->second;
    }

    auto secret_key_it = custom_params->find ("secret_key");
    if (secret_key_it != custom_params->end()) {
        secret_key = secret_key_it->second;
    }

    auto session_token_it = custom_params->find ("session_token");
    if (session_token_it != custom_params->end()) {
        session_token = session_token_it->second;
    }

    if (access_key.empty() || secret_key.empty()) {
        return std::nullopt;
    }

    if (session_token.empty()) {
        return Aws::Auth::AWSCredentials (access_key, secret_key);
    }
    return Aws::Auth::AWSCredentials (access_key, secret_key, session_token);
}

bool
getUserVirtualAddressing (nixl_b_params_t *custom_params) {
    if (!custom_params) {
        return false;
    }

    auto virtual_addressing_it = custom_params->find ("use_virtual_addressing");
    if (virtual_addressing_it != custom_params->end()) {
        const std::string &value = virtual_addressing_it->second;
        if (value == "true") {
            return true;
        } else if (value == "false") {
            return false;
        } else {
            throw std::runtime_error ("Invalid value for use_virtual_addressing: '" + value +
                                      "'. Must be 'true' or 'false'");
        }
    }

    return false;
}

std::string
getBucketName (nixl_b_params_t *custom_params) {
    if (custom_params) {
        auto bucket_it = custom_params->find ("bucket");
        if (bucket_it != custom_params->end() && !bucket_it->second.empty()) {
            return bucket_it->second;
        }
    }

    const char *env_bucket = std::getenv ("AWS_DEFAULT_BUCKET");
    if (env_bucket && env_bucket[0] != '\0') {
        return std::string (env_bucket);
    }

    throw std::runtime_error ("Bucket name not found. Please provide 'bucket' in custom_params or "
                              "set AWS_DEFAULT_BUCKET environment variable");
}
} // namespace

AwsS3Client::AwsS3Client (nixl_b_params_t *custom_params,
                          std::shared_ptr<Aws::Utils::Threading::Executor> executor) :
        aws_options_ (
                []() {
                    auto *opts = new Aws::SDKOptions();
                    Aws::InitAPI (*opts);
                    return opts;
                }(),
                [] (Aws::SDKOptions *opts) {
                    Aws::ShutdownAPI (*opts);
                    delete opts;
                }) {
    auto config = ::createClientConfiguration (custom_params);
    if (executor) {
        config.executor = executor;
    }
    auto credentials_opt = ::createAWSCredentials (custom_params);
    bool use_virtual_addressing = ::getUserVirtualAddressing (custom_params);
    bucket_name_ = Aws::String(::getBucketName (custom_params));

    if (credentials_opt.has_value()) {
        s3_client_ = std::make_unique<Aws::S3::S3Client> (
                credentials_opt.value(),
                config,
                Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::RequestDependent,
                use_virtual_addressing);
    } else {
        s3_client_ = std::make_unique<Aws::S3::S3Client> (
                config,
                Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::RequestDependent,
                use_virtual_addressing);
    }
}

void
AwsS3Client::setExecutor(std::shared_ptr<Aws::Utils::Threading::Executor> executor) {
    throw std::runtime_error("AwsS3Client::setExecutor() not supported - AWS SDK doesn't allow changing executor after client creation");
}

void
AwsS3Client::PutObjectAsync (std::string_view key,
                             uintptr_t data_ptr,
                             size_t data_len,
                             size_t offset,
                             PutObjectCallback callback) {
    // AWS S3 doesn't support partial put operations with offset
    if (offset != 0) {
        callback(false);
        return;
    }

    Aws::S3::Model::PutObjectRequest request;
    request.WithBucket (bucket_name_).WithKey (Aws::String (key));

    auto preallocated_stream_buf = Aws::MakeShared<Aws::Utils::Stream::PreallocatedStreamBuf>(
            "PutObjectStreamBuf",
            reinterpret_cast<unsigned char*>(data_ptr),
            data_len);
    auto data_stream = Aws::MakeShared<Aws::IOStream>("PutObjectInputStream", preallocated_stream_buf.get());
    request.SetBody(data_stream);

    s3_client_->PutObjectAsync (
            request,
            [callback, preallocated_stream_buf, data_stream] (const Aws::S3::S3Client *client,
                        const Aws::S3::Model::PutObjectRequest &req,
                        const Aws::S3::Model::PutObjectOutcome &outcome,
                        const std::shared_ptr<const Aws::Client::AsyncCallerContext> &context) {
                callback (outcome.IsSuccess());
            },
            nullptr);
}

void
AwsS3Client::GetObjectAsync (std::string_view key,
                             uintptr_t data_ptr,
                             size_t data_len,
                             size_t offset,
                             GetObjectCallback callback) {
    Aws::S3::Model::GetObjectRequest request;
    request.WithBucket (bucket_name_).WithKey (Aws::String (key));

    if (offset > 0) {
        request.SetRange(absl::StrFormat("bytes=%d-%d", offset, offset + data_len - 1));
    }

    s3_client_->GetObjectAsync (
            request,
            [callback, data_ptr, data_len] (
                    const Aws::S3::S3Client *client,
                    const Aws::S3::Model::GetObjectRequest &req,
                    const Aws::S3::Model::GetObjectOutcome &outcome,
                    const std::shared_ptr<const Aws::Client::AsyncCallerContext> &context) {
                if (outcome.IsSuccess()) {
                    auto &stream = outcome.GetResult().GetBody();
                    stream.seekg (0, std::ios::beg);
                    stream.read (reinterpret_cast<char *> (data_ptr), data_len);
                }
                callback (outcome.IsSuccess());
            },
            nullptr);
}
