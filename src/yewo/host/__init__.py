"""Every host difference Yewo adapts to: locks, durability, tools, processes."""

from yewo.host.executables import (
    ANDROID_SDK_ENVIRONMENT,
    build_host_command,
    default_android_sdk_root,
    executable_suffixes,
    is_node_script_shim,
    node_executable_patterns,
    node_installation_roots,
    resolve_android_sdk_root,
    resolve_android_sdk_tool,
    resolve_build_tools_executables,
    resolve_emulator_executable,
    resolve_host_executable,
    resolve_node_package_entry_point,
    resolve_sdk_executable,
    sorted_existing_paths,
)
from yewo.host.platform_support import (
    supports_android_emulator_clipboard_control,
    supports_apple_device_tooling,
)
from yewo.host.process_control import (
    ManagedProcess,
    start_detached_process,
    stop_process_tree,
)
from yewo.host.file_durability import flush_directory_entry, replace_file_durably
from yewo.host.file_locking import (
    LOCK_FILE_MODE,
    LOCK_FILE_SUFFIX,
    exclusive_file_lock,
)

__all__ = [
    "ANDROID_SDK_ENVIRONMENT",
    "LOCK_FILE_MODE",
    "ManagedProcess",
    "LOCK_FILE_SUFFIX",
    "build_host_command",
    "default_android_sdk_root",
    "exclusive_file_lock",
    "executable_suffixes",
    "flush_directory_entry",
    "is_node_script_shim",
    "node_executable_patterns",
    "node_installation_roots",
    "replace_file_durably",
    "resolve_android_sdk_root",
    "resolve_android_sdk_tool",
    "resolve_build_tools_executables",
    "resolve_emulator_executable",
    "resolve_host_executable",
    "resolve_node_package_entry_point",
    "resolve_sdk_executable",
    "sorted_existing_paths",
    "start_detached_process",
    "stop_process_tree",
    "supports_android_emulator_clipboard_control",
    "supports_apple_device_tooling",
]
