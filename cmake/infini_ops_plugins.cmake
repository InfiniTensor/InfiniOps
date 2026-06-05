include_guard(GLOBAL)

set(INFINI_OPS_PLUGINS "" CACHE STRING
    "Comma- or semicolon-separated `infini_ops` build-time plugins to enable.")
set(INFINI_OPS_PLUGIN_ROOT "${PROJECT_SOURCE_DIR}/plugins" CACHE PATH
    "Directory containing built-in `infini_ops` build-time plugins.")
set(INFINI_OPS_PLUGIN_ROOTS "" CACHE STRING
    "Additional comma- or semicolon-separated `infini_ops` build-time plugin roots.")
set(INFINI_OPS_PLUGIN_CONTRACT_VERSION 1)

set(_INFINI_OPS_KNOWN_DEVICE_PLUGINS
    cpu nvidia iluvatar hygon metax moore cambricon ascend)

function(_infini_ops_get_plugin_roots out_var)
    set(_roots "${INFINI_OPS_PLUGIN_ROOT}")

    if(INFINI_OPS_PLUGIN_ROOTS)
        set(_extra_roots "${INFINI_OPS_PLUGIN_ROOTS}")
        string(REPLACE "," ";" _extra_roots "${_extra_roots}")
        foreach(_root IN LISTS _extra_roots)
            string(STRIP "${_root}" _root)
            if(NOT _root STREQUAL "")
                list(APPEND _roots "${_root}")
            endif()
        endforeach()
    endif()

    if(_roots)
        list(REMOVE_DUPLICATES _roots)
    endif()

    set(${out_var} ${_roots} PARENT_SCOPE)
endfunction()

function(_infini_ops_find_plugin_manifest name out_var)
    _infini_ops_get_plugin_roots(_plugin_roots)
    set(_matches)

    foreach(_root IN LISTS _plugin_roots)
        set(_candidate "${_root}/${name}/plugin.json")
        if(EXISTS "${_candidate}")
            list(APPEND _matches "${_candidate}")
        endif()
    endforeach()

    list(LENGTH _matches _match_count)
    if(_match_count EQUAL 0)
        string(REPLACE ";" "`, `" _roots_message "${_plugin_roots}")
        message(FATAL_ERROR
            "`infini_ops` plugin `${name}` manifest `plugin.json` was not found in roots: "
            "`${_roots_message}`.")
    elseif(_match_count GREATER 1)
        string(REPLACE ";" "`, `" _matches_message "${_matches}")
        message(FATAL_ERROR
            "`infini_ops` plugin `${name}` has duplicate manifests: "
            "`${_matches_message}`.")
    endif()

    list(GET _matches 0 _manifest_path)
    set(${out_var} "${_manifest_path}" PARENT_SCOPE)
endfunction()

function(_infini_ops_read_manifest_cmake_entry manifest_path out_var)
    file(READ "${manifest_path}" _manifest_json)
    string(REGEX MATCH
        "\"cmake_entry\"[ \t\r\n]*:[ \t\r\n]*\"([^\"]+)\""
        _cmake_entry_match
        "${_manifest_json}")

    if(NOT _cmake_entry_match)
        message(FATAL_ERROR
            "`infini_ops` plugin manifest `${manifest_path}` is missing `cmake_entry`.")
    endif()

    set(${out_var} "${CMAKE_MATCH_1}" PARENT_SCOPE)
endfunction()

function(_infini_ops_find_plugin_entry name out_var)
    _infini_ops_find_plugin_manifest("${name}" _manifest_path)
    _infini_ops_read_manifest_cmake_entry("${_manifest_path}" _cmake_entry)

    if(IS_ABSOLUTE "${_cmake_entry}" OR _cmake_entry MATCHES "(^|/)\\.\\.(/|$)")
        message(FATAL_ERROR
            "`infini_ops` plugin `${name}` field `cmake_entry` must be a relative "
            "path inside the plugin directory.")
    endif()

    get_filename_component(_plugin_dir "${_manifest_path}" DIRECTORY)
    set(_entry_path "${_plugin_dir}/${_cmake_entry}")
    if(NOT EXISTS "${_entry_path}")
        message(FATAL_ERROR
            "`infini_ops` plugin `${name}` `CMake` entry was not found: `${_entry_path}`.")
    endif()

    set(${out_var} "${_entry_path}" PARENT_SCOPE)
endfunction()

function(_infini_ops_append_unique_global property_name)
    get_property(_values GLOBAL PROPERTY "${property_name}")
    foreach(_value ${ARGN})
        if("${_value}" STREQUAL "")
            continue()
        endif()

        list(FIND _values "${_value}" _index)
        if(_index EQUAL -1)
            set_property(GLOBAL APPEND PROPERTY "${property_name}" "${_value}")
        endif()
    endforeach()
endfunction()

function(infini_ops_register_plugin)
    set(_one_value_args NAME KIND CONTRACT_VERSION CMAKE_ENTRY)
    set(_multi_value_args
        DEVICES
        DEPENDS
        SOURCE_ROOTS
        OPERATOR_ROOTS
        DEVICE_HEADERS
        TEST_DEVICES)
    cmake_parse_arguments(ARG "" "${_one_value_args}" "${_multi_value_args}" ${ARGN})

    foreach(_required NAME KIND CONTRACT_VERSION CMAKE_ENTRY)
        if(NOT ARG_${_required})
            message(FATAL_ERROR "`infini_ops_register_plugin` is missing `${_required}`.")
        endif()
    endforeach()

    if(NOT ARG_KIND STREQUAL "shared" AND NOT ARG_KIND STREQUAL "device")
        message(FATAL_ERROR "`infini_ops` plugin `${ARG_NAME}` has invalid `kind`: `${ARG_KIND}`.")
    endif()

    if(NOT "${ARG_CONTRACT_VERSION}" STREQUAL "${INFINI_OPS_PLUGIN_CONTRACT_VERSION}")
        message(FATAL_ERROR
            "`infini_ops` plugin `${ARG_NAME}` uses contract `${ARG_CONTRACT_VERSION}`; "
            "expected `${INFINI_OPS_PLUGIN_CONTRACT_VERSION}`.")
    endif()

    foreach(_device IN LISTS ARG_DEVICES)
        list(FIND _INFINI_OPS_KNOWN_DEVICE_PLUGINS "${_device}" _known_index)
        if(_known_index EQUAL -1)
            message(FATAL_ERROR
                "`infini_ops` plugin `${ARG_NAME}` declares unknown device `${_device}`.")
        endif()
    endforeach()

    if(ARG_KIND STREQUAL "device" AND NOT ARG_DEVICES)
        message(FATAL_ERROR "`infini_ops` device plugin `${ARG_NAME}` must declare `DEVICES`.")
    endif()

    if(ARG_KIND STREQUAL "shared" AND ARG_DEVICES)
        message(FATAL_ERROR "`infini_ops` shared plugin `${ARG_NAME}` must not declare `DEVICES`.")
    endif()

    _infini_ops_append_unique_global(INFINI_OPS_PLUGIN_NAMES "${ARG_NAME}")
    _infini_ops_append_unique_global(INFINI_OPS_PLUGIN_DEVICES ${ARG_DEVICES})
    _infini_ops_append_unique_global(INFINI_OPS_PLUGIN_SOURCE_ROOTS ${ARG_SOURCE_ROOTS})
    _infini_ops_append_unique_global(INFINI_OPS_PLUGIN_OPERATOR_ROOTS ${ARG_OPERATOR_ROOTS})
    _infini_ops_append_unique_global(INFINI_OPS_PLUGIN_DEVICE_HEADERS ${ARG_DEVICE_HEADERS})
    _infini_ops_append_unique_global(INFINI_OPS_PLUGIN_TEST_DEVICES ${ARG_TEST_DEVICES})
endfunction()

function(infini_ops_register_device)
    infini_ops_register_plugin(
        KIND device
        CONTRACT_VERSION ${INFINI_OPS_PLUGIN_CONTRACT_VERSION}
        ${ARGN})
endfunction()

function(infini_ops_enable_plugin name)
    get_property(_loaded GLOBAL PROPERTY INFINI_OPS_PLUGIN_LOADED)
    list(FIND _loaded "${name}" _loaded_index)
    if(NOT _loaded_index EQUAL -1)
        return()
    endif()

    get_property(_loading GLOBAL PROPERTY INFINI_OPS_PLUGIN_LOADING_STACK)
    list(FIND _loading "${name}" _loading_index)
    if(NOT _loading_index EQUAL -1)
        list(APPEND _loading "${name}")
        string(REPLACE ";" " -> " _cycle "${_loading}")
        message(FATAL_ERROR "`infini_ops` plugin dependency cycle detected: `${_cycle}`.")
    endif()

    _infini_ops_find_plugin_entry("${name}" _entry_path)

    set_property(GLOBAL APPEND PROPERTY INFINI_OPS_PLUGIN_LOADING_STACK "${name}")
    include("${_entry_path}")
    get_property(_loading GLOBAL PROPERTY INFINI_OPS_PLUGIN_LOADING_STACK)
    list(REMOVE_ITEM _loading "${name}")
    set_property(GLOBAL PROPERTY INFINI_OPS_PLUGIN_LOADING_STACK "${_loading}")

    get_property(_registered GLOBAL PROPERTY INFINI_OPS_PLUGIN_NAMES)
    list(FIND _registered "${name}" _registered_index)
    if(_registered_index EQUAL -1)
        message(FATAL_ERROR "`infini_ops` plugin `${name}` did not call `infini_ops_register_plugin`.")
    endif()

    set_property(GLOBAL APPEND PROPERTY INFINI_OPS_PLUGIN_LOADED "${name}")
endfunction()

function(infini_ops_enable_requested_plugins)
    set(_requested)

    if(INFINI_OPS_PLUGINS)
        set(_raw_plugins "${INFINI_OPS_PLUGINS}")
        string(REPLACE "," ";" _raw_plugins "${_raw_plugins}")
        foreach(_plugin IN LISTS _raw_plugins)
            string(STRIP "${_plugin}" _plugin)
            if(NOT _plugin STREQUAL "")
                list(APPEND _requested "${_plugin}")
            endif()
        endforeach()
    endif()

    if(WITH_CPU)
        list(APPEND _requested cpu)
    endif()
    if(WITH_NVIDIA)
        list(APPEND _requested nvidia)
    endif()
    if(WITH_ILUVATAR)
        list(APPEND _requested iluvatar)
    endif()
    if(WITH_HYGON)
        list(APPEND _requested hygon)
    endif()
    if(WITH_METAX)
        list(APPEND _requested metax)
    endif()
    if(WITH_MOORE)
        list(APPEND _requested moore)
    endif()
    if(WITH_CAMBRICON)
        list(APPEND _requested cambricon)
    endif()
    if(WITH_ASCEND)
        list(APPEND _requested ascend)
    endif()

    if(_requested)
        list(REMOVE_DUPLICATES _requested)
    else()
        list(APPEND _requested cpu)
        set(WITH_CPU ON CACHE BOOL "Enable CPU backend" FORCE)
    endif()

    foreach(_plugin IN LISTS _requested)
        infini_ops_enable_plugin("${_plugin}")
    endforeach()
endfunction()

function(infini_ops_get_enabled_devices out_var)
    get_property(_devices GLOBAL PROPERTY INFINI_OPS_PLUGIN_DEVICES)
    if(NOT _devices)
        set(_devices)
    endif()
    set(${out_var} ${_devices} PARENT_SCOPE)
endfunction()

function(_infini_ops_json_escape value out_var)
    string(REPLACE "\\" "\\\\" _escaped "${value}")
    string(REPLACE "\"" "\\\"" _escaped "${_escaped}")
    set(${out_var} "${_escaped}" PARENT_SCOPE)
endfunction()

function(_infini_ops_append_json_array path field trailing_comma)
    file(APPEND "${path}" "  \"${field}\": [")
    set(_first TRUE)
    foreach(_value ${ARGN})
        if(_first)
            set(_first FALSE)
        else()
            file(APPEND "${path}" ", ")
        endif()
        _infini_ops_json_escape("${_value}" _escaped)
        file(APPEND "${path}" "\"${_escaped}\"")
    endforeach()
    file(APPEND "${path}" "]")
    if(trailing_comma)
        file(APPEND "${path}" ",")
    endif()
    file(APPEND "${path}" "\n")
endfunction()

function(_infini_ops_append_json_map path field trailing_comma)
    file(APPEND "${path}" "  \"${field}\": {")
    set(_first TRUE)
    foreach(_entry ${ARGN})
        string(FIND "${_entry}" "=" _equals)
        if(_equals EQUAL -1)
            message(FATAL_ERROR "Invalid `infini_ops` plugin map entry `${_entry}`.")
        endif()
        string(SUBSTRING "${_entry}" 0 ${_equals} _key)
        math(EXPR _value_start "${_equals} + 1")
        string(SUBSTRING "${_entry}" ${_value_start} -1 _value)

        if(_first)
            set(_first FALSE)
        else()
            file(APPEND "${path}" ",")
        endif()
        _infini_ops_json_escape("${_key}" _escaped_key)
        _infini_ops_json_escape("${_value}" _escaped_value)
        file(APPEND "${path}" "\n    \"${_escaped_key}\": \"${_escaped_value}\"")
    endforeach()
    if(NOT _first)
        file(APPEND "${path}" "\n  ")
    endif()
    file(APPEND "${path}" "}")
    if(trailing_comma)
        file(APPEND "${path}" ",")
    endif()
    file(APPEND "${path}" "\n")
endfunction()

function(infini_ops_write_plugin_registry path)
    get_property(_plugins GLOBAL PROPERTY INFINI_OPS_PLUGIN_NAMES)
    get_property(_devices GLOBAL PROPERTY INFINI_OPS_PLUGIN_DEVICES)
    get_property(_source_roots GLOBAL PROPERTY INFINI_OPS_PLUGIN_SOURCE_ROOTS)
    get_property(_operator_roots GLOBAL PROPERTY INFINI_OPS_PLUGIN_OPERATOR_ROOTS)
    get_property(_device_headers GLOBAL PROPERTY INFINI_OPS_PLUGIN_DEVICE_HEADERS)
    get_property(_test_devices GLOBAL PROPERTY INFINI_OPS_PLUGIN_TEST_DEVICES)

    file(WRITE "${path}" "{\n")
    _infini_ops_append_json_array("${path}" "plugins" TRUE ${_plugins})
    _infini_ops_append_json_array("${path}" "devices" TRUE ${_devices})
    _infini_ops_append_json_array("${path}" "source_roots" TRUE ${_source_roots})
    _infini_ops_append_json_array("${path}" "operator_roots" TRUE ${_operator_roots})
    _infini_ops_append_json_map("${path}" "device_headers" TRUE ${_device_headers})
    _infini_ops_append_json_map("${path}" "test_devices" FALSE ${_test_devices})
    file(APPEND "${path}" "}\n")

    message(STATUS "`infini_ops` plugins: `${_plugins}`.")
    message(STATUS "`infini_ops` plugin devices: `${_devices}`.")
endfunction()
