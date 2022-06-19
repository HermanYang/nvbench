DROP VIEW IF EXISTS step_range;
CREATE VIEW step_range AS 
    SELECT 
        global_tid_map(NVTX_EVENTS.globalTid) AS card_id,
        NVTX_EVENTS.globalTid, 
        NVTX_EVENTS.text,
        NVTX_EVENTS.start, 
        NVTX_EVENTS.end, 
        (NVTX_EVENTS.end - NVTX_EVENTS.start) AS duration
    FROM NVTX_EVENTS 
    WHERE 
        nvtx_event_map(eventType) == "NvtxPushPopRange" AND 
        NVTX_EVENTS.text LIKE "step%";

DROP VIEW IF EXISTS nvtx_range;
CREATE VIEW nvtx_range AS 
    SELECT 
        step_range.card_id,
        NVTX_EVENTS.globalTid, 
        step_map(step_range.text) AS step_id, 
        NVTX_EVENTS.start, 
        NVTX_EVENTS.end, 
        (NVTX_EVENTS.end - NVTX_EVENTS.start) AS duration,
        NVTX_EVENTS.category AS category_id, 
        NVTX_EVENTS.text AS range_name
    FROM NVTX_EVENTS 
        JOIN step_range 
        ON NVTX_EVENTS.globalTid == step_range.globalTid 
        WHERE 
            NVTX_EVENTS.start > step_range.start AND 
            NVTX_EVENTS.end < step_range.end AND 
            nvtx_event_map(eventType) == "NvtxPushPopRange" AND 
            NVTX_EVENTS.text IS NOT NULL AND 
            NVTX_EVENTS.text != "";

DROP VIEW IF EXISTS input_range;
CREATE VIEW input_range AS 
    SELECT 
        nvtx_range.card_id, 
        nvtx_range.globalTid, 
        nvtx_range.step_id, 
        nvtx_range.start, 
        nvtx_range.end, 
        nvtx_range.duration,
        category_map(nvtx_range.category_id) AS range_category, 
        nvtx_range.range_name
    FROM nvtx_range
    WHERE 
        category_map(nvtx_range.category_id) == "input";

DROP VIEW IF EXISTS compute_range;
CREATE VIEW compute_range AS 
    SELECT 
        nvtx_range.card_id, 
        nvtx_range.globalTid, 
        nvtx_range.step_id, 
        nvtx_range.start, 
        nvtx_range.end, 
        nvtx_range.duration,
        category_map(nvtx_range.category_id) AS range_category, 
        nvtx_range.range_name
    FROM nvtx_range 
    WHERE category_map(nvtx_range.category_id) == "compute";

DROP VIEW IF EXISTS communication_range;
CREATE VIEW communication_range AS 
    SELECT 
        nvtx_range.card_id, 
        nvtx_range.globalTid, 
        nvtx_range.step_id, 
        nvtx_range.start, 
        nvtx_range.end, 
        nvtx_range.duration,
        category_map(nvtx_range.category_id) AS range_category, 
        nvtx_range.range_name
    FROM nvtx_range 
    WHERE 
        nvtx_range.range_name == "all to all" AND
        category_map(nvtx_range.category_id) == "communication";

DROP VIEW IF EXISTS output_range;
CREATE VIEW output_range AS 
    SELECT 
        nvtx_range.card_id, 
        nvtx_range.globalTid, 
        nvtx_range.step_id, 
        nvtx_range.start, 
        nvtx_range.end, 
        nvtx_range.duration,
        category_map(nvtx_range.category_id) AS range_category, 
        nvtx_range.range_name
    FROM nvtx_range 
    WHERE category_map(nvtx_range.category_id) == "output";

DROP VIEW IF EXISTS inference_range;
CREATE VIEW inference_range AS 
    SELECT 
        nvtx_range.card_id, 
        nvtx_range.globalTid, 
        nvtx_range.step_id, 
        nvtx_range.start, 
        nvtx_range.end, 
        nvtx_range.duration,
        category_map(nvtx_range.category_id) AS range_category, 
        nvtx_range.range_name
    FROM nvtx_range 
    WHERE category_map(nvtx_range.category_id) == "inference";

DROP VIEW IF EXISTS input_runtime;
CREATE VIEW input_runtime AS 
    SELECT 
    CUPTI_ACTIVITY_KIND_RUNTIME.globalTid, 
    CUPTI_ACTIVITY_KIND_RUNTIME.start, 
    CUPTI_ACTIVITY_KIND_RUNTIME.end, 
    CUPTI_ACTIVITY_KIND_RUNTIME.correlationId, 
    string_map(CUPTI_ACTIVITY_KIND_RUNTIME.nameId) AS function_name, 
    input_range.card_id, 
    input_range.step_id, 
    input_range.range_category,
    input_range.range_name
    FROM CUPTI_ACTIVITY_KIND_RUNTIME 
    JOIN input_range 
    ON 
        CUPTI_ACTIVITY_KIND_RUNTIME.globalTid == input_range.globalTid AND 
        CUPTI_ACTIVITY_KIND_RUNTIME.start > input_range.start AND 
        CUPTI_ACTIVITY_KIND_RUNTIME.end < input_range.end AND 
        function_name LIKE "%Memcpy%";

DROP VIEW IF EXISTS compute_runtime;
CREATE VIEW compute_runtime AS 
    SELECT 
        CUPTI_ACTIVITY_KIND_RUNTIME.globalTid, 
        CUPTI_ACTIVITY_KIND_RUNTIME.start,
        CUPTI_ACTIVITY_KIND_RUNTIME.end,
        CUPTI_ACTIVITY_KIND_RUNTIME.correlationId,
        string_map(CUPTI_ACTIVITY_KIND_RUNTIME.nameId) AS function_name, 
        compute_range.card_id, 
        compute_range.step_id,
        compute_range.range_category,
        compute_range.range_name
    FROM CUPTI_ACTIVITY_KIND_RUNTIME 
        JOIN compute_range 
        ON 
            CUPTI_ACTIVITY_KIND_RUNTIME.globalTid == compute_range.globalTid AND 
            CUPTI_ACTIVITY_KIND_RUNTIME.start > compute_range.start AND 
            CUPTI_ACTIVITY_KIND_RUNTIME.end < compute_range.end AND
            function_name LIKE "%LaunchKernel%";

DROP VIEW IF EXISTS communication_runtime;
CREATE VIEW communication_runtime AS 
    SELECT 
        CUPTI_ACTIVITY_KIND_RUNTIME.globalTid, 
        CUPTI_ACTIVITY_KIND_RUNTIME.start,
        CUPTI_ACTIVITY_KIND_RUNTIME.end,
        CUPTI_ACTIVITY_KIND_RUNTIME.correlationId,
        string_map(CUPTI_ACTIVITY_KIND_RUNTIME.nameId) AS function_name, 
        communication_range.card_id,
        communication_range.step_id,
        communication_range.range_category,
        communication_range.range_name
    FROM CUPTI_ACTIVITY_KIND_RUNTIME 
        JOIN communication_range 
        ON 
            CUPTI_ACTIVITY_KIND_RUNTIME.globalTid == communication_range.globalTid AND 
            CUPTI_ACTIVITY_KIND_RUNTIME.start > communication_range.start AND 
            CUPTI_ACTIVITY_KIND_RUNTIME.end < communication_range.end AND
            function_name LIKE "%LaunchKernel%";

DROP VIEW IF EXISTS output_runtime;
CREATE VIEW output_runtime AS 
    SELECT 
        CUPTI_ACTIVITY_KIND_RUNTIME.globalTid, 
        CUPTI_ACTIVITY_KIND_RUNTIME.start,
        CUPTI_ACTIVITY_KIND_RUNTIME.end,
        CUPTI_ACTIVITY_KIND_RUNTIME.correlationId,
        string_map(CUPTI_ACTIVITY_KIND_RUNTIME.nameId) AS function_name, 
        output_range.card_id,
        output_range.step_id,
        output_range.range_category,
        output_range.range_name
    FROM CUPTI_ACTIVITY_KIND_RUNTIME 
        JOIN output_range 
        ON 
            CUPTI_ACTIVITY_KIND_RUNTIME.globalTid == output_range.globalTid AND 
            CUPTI_ACTIVITY_KIND_RUNTIME.start > output_range.start AND 
            CUPTI_ACTIVITY_KIND_RUNTIME.end < output_range.end AND
            function_name LIKE "%Memcpy%";

DROP VIEW IF EXISTS input_kernel;
CREATE VIEW input_kernel AS 
    SELECT 
        input_runtime.card_id,
        input_runtime.globalTid,
        input_runtime.step_id,
        input_runtime.range_category,
        input_runtime.function_name,
        input_runtime.range_name,
        CUPTI_ACTIVITY_KIND_MEMCPY.bytes,
        (CUPTI_ACTIVITY_KIND_MEMCPY.end - CUPTI_ACTIVITY_KIND_MEMCPY.start) AS duration,
        ((bytes * 1.0)/((CUPTI_ACTIVITY_KIND_MEMCPY.end - CUPTI_ACTIVITY_KIND_MEMCPY.start) * 0.000000001)) AS throughput 
    FROM input_runtime 
    JOIN CUPTI_ACTIVITY_KIND_MEMCPY 
        ON 
            input_runtime.correlationId == CUPTI_ACTIVITY_KIND_MEMCPY.correlationId;

DROP VIEW IF EXISTS communication_kernel;
CREATE VIEW communication_kernel AS 
    SELECT 
        communication_runtime.card_id,
        communication_runtime.globalTid,
        communication_runtime.step_id,
        communication_runtime.range_category,
        communication_runtime.range_name,
        (CUPTI_ACTIVITY_KIND_KERNEL.end - CUPTI_ACTIVITY_KIND_KERNEL.start) AS duration 
    FROM communication_runtime 
        JOIN CUPTI_ACTIVITY_KIND_KERNEL ON communication_runtime.correlationId == CUPTI_ACTIVITY_KIND_KERNEL.correlationId;

DROP VIEW IF EXISTS compute_kernel;
CREATE VIEW compute_kernel AS 
    SELECT 
        compute_runtime.card_id,
        compute_runtime.globalTid,
        compute_runtime.step_id,
        compute_runtime.range_category,
        compute_runtime.range_name,
        compute_runtime.function_name,
        (CUPTI_ACTIVITY_KIND_KERNEL.end - CUPTI_ACTIVITY_KIND_KERNEL.start) AS duration 
    FROM compute_runtime 
        JOIN CUPTI_ACTIVITY_KIND_KERNEL 
        ON compute_runtime.correlationId == CUPTI_ACTIVITY_KIND_KERNEL.correlationId;

DROP VIEW IF EXISTS bottom_mlp_kernel;
CREATE VIEW bottom_mlp_kernel AS 
    SELECT * FROM compute_kernel WHERE range_name LIKE "bottom mlp";

DROP VIEW IF EXISTS embedding_kernel;
CREATE VIEW embedding_kernel AS 
    SELECT * FROM compute_kernel WHERE range_name LIKE "embedding";

DROP VIEW IF EXISTS interaction_kernel;
CREATE VIEW interaction_kernel AS 
    SELECT * FROM compute_kernel WHERE range_name LIKE "interaction";

DROP VIEW IF EXISTS top_mlp_kernel;
CREATE VIEW top_mlp_kernel AS 
    SELECT * FROM compute_kernel WHERE range_name LIKE "top mlp";

DROP VIEW IF EXISTS output_kernel;
CREATE VIEW output_kernel AS 
    SELECT 
        output_runtime.card_id,
        output_runtime.globalTid,
        output_runtime.step_id,
        output_runtime.range_category,
        output_runtime.range_name,
        CUPTI_ACTIVITY_KIND_MEMCPY.bytes,
        (CUPTI_ACTIVITY_KIND_MEMCPY.end - CUPTI_ACTIVITY_KIND_MEMCPY.start) AS duration,
        ((bytes * 1.0)/((CUPTI_ACTIVITY_KIND_MEMCPY.end - CUPTI_ACTIVITY_KIND_MEMCPY.start) * 0.000000001)) AS throughput 
    FROM output_runtime 
    JOIN CUPTI_ACTIVITY_KIND_MEMCPY ON output_runtime.correlationId == CUPTI_ACTIVITY_KIND_MEMCPY.correlationId;

DROP VIEW IF EXISTS input_device_stats;
CREATE VIEW input_device_stats AS 
    SELECT 
        input_kernel.card_id,
        input_kernel.globalTid,
        input_kernel.step_id,
        input_kernel.range_category,
        input_kernel.range_name,
        SUM(input_kernel.bytes) AS bytes, 
        SUM(input_kernel.duration) AS duration,
        AVG(input_kernel.throughput) AS throughput
    FROM input_kernel
    GROUP BY 
        card_id,
        input_kernel.step_id,
        input_kernel.range_category,
        input_kernel.range_name;

DROP VIEW IF EXISTS bottom_mlp_device_stats;
CREATE VIEW bottom_mlp_device_stats AS 
    SELECT 
        bottom_mlp_kernel.card_id,
        bottom_mlp_kernel.globalTid, 
        bottom_mlp_kernel.step_id,
        bottom_mlp_kernel.range_category,
        bottom_mlp_kernel.range_name,
        SUM(bottom_mlp_kernel.duration) AS duration
    FROM bottom_mlp_kernel
    GROUP BY 
        card_id,
        bottom_mlp_kernel.step_id;

DROP VIEW IF EXISTS embedding_device_stats;
CREATE VIEW embedding_device_stats AS 
    SELECT 
        embedding_kernel.card_id,
        embedding_kernel.globalTid, 
        embedding_kernel.step_id,
        embedding_kernel.range_category,
        embedding_kernel.range_name,
        SUM(embedding_kernel.duration) AS duration
    FROM embedding_kernel
    GROUP BY 
        card_id,
        embedding_kernel.step_id;

DROP VIEW IF EXISTS interaction_device_stats;
CREATE VIEW interaction_device_stats AS 
    SELECT 
        interaction_kernel.card_id,
        interaction_kernel.globalTid, 
        interaction_kernel.step_id,
        interaction_kernel.range_category,
        interaction_kernel.range_name,
        SUM(interaction_kernel.duration) AS duration
    FROM interaction_kernel
    GROUP BY 
        card_id,
        interaction_kernel.step_id;

DROP VIEW IF EXISTS top_mlp_device_stats;
CREATE VIEW top_mlp_device_stats AS 
    SELECT 
        top_mlp_kernel.card_id,
        top_mlp_kernel.globalTid, 
        top_mlp_kernel.step_id,
        top_mlp_kernel.range_category,
        top_mlp_kernel.range_name,
        SUM(top_mlp_kernel.duration) AS duration
    FROM top_mlp_kernel
    GROUP BY 
        card_id,
        top_mlp_kernel.step_id;

DROP VIEW IF EXISTS communication_device_stats;
CREATE VIEW communication_device_stats AS 
    SELECT 
        communication_kernel.card_id,
        communication_kernel.globalTid,
        communication_kernel.step_id,
        communication_kernel.range_category,
        communication_kernel.range_name,
        SUM(communication_kernel.duration) AS duration
    FROM communication_kernel 
    GROUP BY 
        card_id,
        communication_kernel.step_id,
        communication_kernel.range_category,
        communication_kernel.range_name;

DROP VIEW IF EXISTS output_device_stats;
CREATE VIEW output_device_stats AS 
    SELECT 
        output_kernel.card_id,
        output_kernel.globalTid,
        output_kernel.step_id,
        output_kernel.range_category,
        output_kernel.range_name,
        SUM(output_kernel.bytes) AS bytes,
        SUM(output_kernel.duration) AS duration,
        AVG(output_kernel.throughput) AS throughput
    FROM output_kernel 
    GROUP BY 
        card_id,
        output_kernel.step_id,
        output_kernel.range_category,
        output_kernel.range_name;

DROP VIEW IF EXISTS input_stats;
CREATE VIEW input_stats AS 
    SELECT 
        input_range.card_id,
        input_range.globalTid,
        input_range.step_id, 
        input_range.duration AS host_duration, 
        input_device_stats.duration AS device_duration,
        input_device_stats.bytes,
        input_device_stats.throughput
    FROM input_range
    JOIN input_device_stats
    ON 
        input_range.card_id == input_device_stats.card_id AND
        input_range.step_id == input_device_stats.step_id AND
        input_range.range_category == input_device_stats.range_category AND
        input_range.range_name == input_device_stats.range_name;

DROP VIEW IF EXISTS bottom_mlp_stats;
CREATE VIEW bottom_mlp_stats AS 
    SELECT 
        compute_range.card_id,
        compute_range.globalTid,
        compute_range.step_id,
        compute_range.duration AS host_duration,
        bottom_mlp_device_stats.duration AS device_duration
    FROM compute_range
    JOIN bottom_mlp_device_stats
    ON 
        compute_range.card_id == bottom_mlp_device_stats.card_id AND
        compute_range.step_id == bottom_mlp_device_stats.step_id AND
        compute_range.range_category == bottom_mlp_device_stats.range_category AND
        compute_range.range_name == bottom_mlp_device_stats.range_name;

DROP VIEW IF EXISTS embedding_stats;
CREATE VIEW embedding_stats AS 
    SELECT 
        compute_range.card_id,
        compute_range.globalTid,
        compute_range.step_id,
        compute_range.duration AS host_duration,
        embedding_device_stats.duration AS device_duration
    FROM compute_range
    JOIN embedding_device_stats
    ON 
        compute_range.card_id == embedding_device_stats.card_id AND
        compute_range.step_id == embedding_device_stats.step_id AND
        compute_range.range_category == embedding_device_stats.range_category AND
        compute_range.range_name == embedding_device_stats.range_name;

DROP VIEW IF EXISTS interaction_stats;
CREATE VIEW interaction_stats AS 
    SELECT 
        compute_range.card_id,
        compute_range.globalTid,
        compute_range.step_id,
        compute_range.duration AS host_duration,
        interaction_device_stats.duration AS device_duration
    FROM compute_range
    JOIN interaction_device_stats
    ON 
        compute_range.card_id == interaction_device_stats.card_id AND
        compute_range.step_id == interaction_device_stats.step_id AND
        compute_range.range_category == interaction_device_stats.range_category AND
        compute_range.range_name == interaction_device_stats.range_name;

DROP VIEW IF EXISTS top_mlp_stats;
CREATE VIEW top_mlp_stats AS 
    SELECT 
        compute_range.card_id,
        compute_range.globalTid,
        compute_range.step_id,
        compute_range.duration AS host_duration,
        top_mlp_device_stats.duration AS device_duration
    FROM compute_range
    JOIN top_mlp_device_stats
    ON 
        compute_range.card_id == top_mlp_device_stats.card_id AND
        compute_range.step_id == top_mlp_device_stats.step_id AND
        compute_range.range_category == top_mlp_device_stats.range_category AND
        compute_range.range_name == top_mlp_device_stats.range_name;

DROP VIEW IF EXISTS communication_stats;
CREATE VIEW communication_stats AS 
    SELECT 
        communication_range.card_id,
        communication_range.globalTid,
        communication_range.step_id,
        communication_range.duration AS host_duration,
        communication_device_stats.duration AS device_duration
    FROM communication_range
    JOIN communication_device_stats
    ON 
        communication_range.card_id == communication_device_stats.card_id AND
        communication_range.step_id == communication_device_stats.step_id AND
        communication_range.range_category == communication_device_stats.range_category AND
        communication_range.range_name == communication_device_stats.range_name;

DROP VIEW IF EXISTS output_stats;
CREATE VIEW output_stats AS 
    SELECT 
        output_range.card_id,
        output_range.globalTid,
        output_range.step_id, 
        output_range.duration AS host_duration, 
        output_device_stats.duration AS device_duration,
        output_device_stats.bytes,
        output_device_stats.throughput
    FROM output_range 
    JOIN output_device_stats
    ON 
        output_range.card_id == output_device_stats.card_id AND
        output_range.step_id == output_device_stats.step_id AND
        output_range.range_category == output_device_stats.range_category AND
        output_range.range_name == output_device_stats.range_name;

DROP VIEW IF EXISTS inference_stats;
CREATE VIEW inference_stats AS 
    SELECT 
        inference_range.card_id,
        inference_range.globalTid,
        inference_range.step_id, 
        inference_range.duration AS host_duration
    FROM inference_range