sh_binary(
    name = "pyclif",
    srcs = ["clif/bin/pyclif"],
    visibility = ["//visibility:public"],
)

sh_binary(
    name = "proto",
    srcs = ["clif/bin/pyclif_proto"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cpp_runtime",
    srcs = glob(
        ["clif/python/*.cc"],
        exclude = ["clif/python/*_test.cc"],
    ),
    hdrs = glob(["clif/python/*.h"]),
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/base:config",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:variant",
        "@com_google_glog//:glog",
        "@com_google_protobuf//:protobuf",
        "@local_config_python//:python_headers",
    ],
)
