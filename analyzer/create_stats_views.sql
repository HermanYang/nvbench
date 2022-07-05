DROP VIEW IF EXISTS step_inference_range;
CREATE VIEW step_inference_range AS 
    SELECT 
        json_extract(NVTX_EVENTS.text, '$.step') as step_id, 
        json_extract(NVTX_EVENTS.text, '$.rank') as rank, 
        json_extract(NVTX_EVENTS.text, '$.local_rank') as local_rank, 
        NVTX_EVENTS.globalTid, 
        NVTX_EVENTS.globalTid / 0x1000000 % 0x1000000 AS pid, 
        NVTX_EVENTS.globalTid % 0x1000000 AS tid,
        json_extract(NVTX_EVENTS.text, '$.event') as event_name,
        json_extract(NVTX_EVENTS.text, '$.name') as event_value,
        json_extract(NVTX_EVENTS.text, '$.batch_size') as batch_size,
        NVTX_EVENTS.start, 
        NVTX_EVENTS.end, 
        (NVTX_EVENTS.end - NVTX_EVENTS.start) AS duration
    FROM NVTX_EVENTS 
    WHERE 
        nvtx_event_map(eventType) == 'NvtxPushPopRange' AND
        json_extract(NVTX_EVENTS.text, '$.event') == 'inference';

DROP VIEW IF EXISTS step_range;
CREATE VIEW step_range AS 
    SELECT 
        step_inference_range.step_id, 
        step_inference_range.rank,
        step_inference_range.local_rank,
        NVTX_EVENTS.globalTid, 
        NVTX_EVENTS.globalTid / 0x1000000 % 0x1000000 AS pid,  
        NVTX_EVENTS.globalTid % 0x1000000 AS tid,
        json_extract(NVTX_EVENTS.text, '$.event') as event_name,
        json_extract(NVTX_EVENTS.text, '$.name') as event_value,
        NVTX_EVENTS.text as event,
        NVTX_EVENTS.start, 
        NVTX_EVENTS.end, 
        (NVTX_EVENTS.end - NVTX_EVENTS.start) AS duration
    FROM NVTX_EVENTS 
    JOIN step_inference_range 
        ON 
            NVTX_EVENTS.globalTid == step_inference_range.globalTid 
    WHERE 
        NVTX_EVENTS.start > step_inference_range.start AND 
        NVTX_EVENTS.end < step_inference_range.end AND 
        nvtx_event_map(eventType) == 'NvtxPushPopRange' AND 
        NVTX_EVENTS.text IS NOT NULL AND 
        NVTX_EVENTS.text != '';

DROP VIEW IF EXISTS step_input_range;
CREATE VIEW step_input_range AS 
    SELECT 
        step_range.step_id, 
        step_range.rank, 
        step_range.local_rank, 
        step_range.globalTid,
        step_range.pid, 
        step_range.tid, 
        step_range.start, 
        step_range.end, 
        step_range.duration,
        step_range.event_name,
        json_extract(event, '$.tensor') as tensor
    FROM step_range
    WHERE 
        step_range.event_name == 'input';

DROP VIEW IF EXISTS step_output_range;
CREATE VIEW step_output_range AS 
    SELECT 
        step_range.step_id, 
        step_range.rank, 
        step_range.local_rank, 
        step_range.globalTid, 
        step_range.pid, 
        step_range.tid, 
        step_range.start, 
        step_range.end, 
        step_range.duration,
        step_range.event_name,
        json_extract(event, '$.tensor') as tensor
    FROM step_range 
    WHERE 
        step_range.event_name == 'output';

DROP VIEW IF EXISTS step_compute_range;
CREATE VIEW step_compute_range AS 
    SELECT 
        step_range.step_id, 
        step_range.rank, 
        step_range.local_rank, 
        step_range.globalTid,
        step_range.pid, 
        step_range.tid, 
        step_range.start, 
        step_range.end, 
        step_range.duration,
        step_range.event_name,
        json_extract(event, '$.name') as event_value,
        json_extract(event, '$.weights') as weights
    FROM step_range 
    WHERE 
        step_range.event_name == 'compute';

DROP VIEW IF EXISTS step_communication_range;
CREATE VIEW step_communication_range AS 
    SELECT 
        step_range.step_id, 
        step_range.rank, 
        step_range.local_rank, 
        step_range.globalTid,
        step_range.pid, 
        step_range.tid, 
        step_range.start, 
        step_range.end, 
        step_range.duration,
        step_range.event_name,
        json_extract(event, '$.name') as event_value,
        json_extract(event, '$.send_tensors') as send_tensors,
        json_extract(event, '$.recv_tensors') as recv_tensors
    FROM step_range 
    WHERE 
        step_range.event_name == 'communication';

DROP VIEW IF EXISTS step_input_runtime;
CREATE VIEW step_input_runtime AS 
    SELECT 
        step_input_range.step_id, 
        step_input_range.rank, 
        step_input_range.local_rank, 
        step_input_range.globalTid,
        step_input_range.pid, 
        step_input_range.tid, 
        step_input_range.event_name,
        CUPTI_ACTIVITY_KIND_RUNTIME.start, 
        CUPTI_ACTIVITY_KIND_RUNTIME.end, 
        CUPTI_ACTIVITY_KIND_RUNTIME.correlationId, 
        string_map(CUPTI_ACTIVITY_KIND_RUNTIME.nameId) AS function_name
        FROM CUPTI_ACTIVITY_KIND_RUNTIME 
    JOIN step_input_range 
    ON 
        CUPTI_ACTIVITY_KIND_RUNTIME.globalTid == step_input_range.globalTid AND 
        CUPTI_ACTIVITY_KIND_RUNTIME.start > step_input_range.start AND 
        CUPTI_ACTIVITY_KIND_RUNTIME.end < step_input_range.end AND 
        function_name LIKE '%Memcpy%';

DROP VIEW IF EXISTS step_compute_runtime;
CREATE VIEW step_compute_runtime AS 
    SELECT 
        step_compute_range.step_id,
        step_compute_range.rank, 
        step_compute_range.local_rank, 
        step_compute_range.globalTid,
        step_compute_range.pid, 
        step_compute_range.tid, 
        step_compute_range.event_name,
        step_compute_range.event_value,
        CUPTI_ACTIVITY_KIND_RUNTIME.start,
        CUPTI_ACTIVITY_KIND_RUNTIME.end,
        CUPTI_ACTIVITY_KIND_RUNTIME.correlationId,
        string_map(CUPTI_ACTIVITY_KIND_RUNTIME.nameId) AS function_name
    FROM CUPTI_ACTIVITY_KIND_RUNTIME 
        JOIN step_compute_range 
        ON 
            CUPTI_ACTIVITY_KIND_RUNTIME.globalTid == step_compute_range.globalTid AND 
            CUPTI_ACTIVITY_KIND_RUNTIME.start > step_compute_range.start AND 
            CUPTI_ACTIVITY_KIND_RUNTIME.end < step_compute_range.end AND
            function_name LIKE '%LaunchKernel%';

DROP VIEW IF EXISTS step_communication_runtime;
CREATE VIEW step_communication_runtime AS 
    SELECT 
        step_communication_range.step_id,
        step_communication_range.rank,
        step_communication_range.local_rank,
        step_communication_range.globalTid, 
        step_communication_range.pid, 
        step_communication_range.tid, 
        step_communication_range.event_name,
        step_communication_range.event_value,
        CUPTI_ACTIVITY_KIND_RUNTIME.start,
        CUPTI_ACTIVITY_KIND_RUNTIME.end,
        CUPTI_ACTIVITY_KIND_RUNTIME.correlationId,
        string_map(CUPTI_ACTIVITY_KIND_RUNTIME.nameId) AS function_name
    FROM CUPTI_ACTIVITY_KIND_RUNTIME 
        JOIN step_communication_range 
        ON 
            CUPTI_ACTIVITY_KIND_RUNTIME.globalTid == step_communication_range.globalTid AND 
            CUPTI_ACTIVITY_KIND_RUNTIME.start > step_communication_range.start AND 
            CUPTI_ACTIVITY_KIND_RUNTIME.end < step_communication_range.end AND
            function_name LIKE '%LaunchKernel%';

DROP VIEW IF EXISTS step_output_runtime;
CREATE VIEW step_output_runtime AS 
    SELECT 
        step_output_range.step_id,
        step_output_range.rank,
        step_output_range.local_rank,
        step_output_range.globalTid, 
        step_output_range.pid, 
        step_output_range.tid, 
        step_output_range.event_name,
        CUPTI_ACTIVITY_KIND_RUNTIME.start,
        CUPTI_ACTIVITY_KIND_RUNTIME.end,
        CUPTI_ACTIVITY_KIND_RUNTIME.correlationId,
        string_map(CUPTI_ACTIVITY_KIND_RUNTIME.nameId) AS function_name
    FROM CUPTI_ACTIVITY_KIND_RUNTIME 
        JOIN step_output_range 
        ON 
            CUPTI_ACTIVITY_KIND_RUNTIME.globalTid == step_output_range.globalTid AND 
            CUPTI_ACTIVITY_KIND_RUNTIME.start > step_output_range.start AND 
            CUPTI_ACTIVITY_KIND_RUNTIME.end < step_output_range.end AND
            function_name LIKE '%Memcpy%';

DROP VIEW IF EXISTS step_input_kernel;
CREATE VIEW step_input_kernel AS 
    SELECT 
        step_input_runtime.step_id,
        step_input_runtime.rank,
        step_input_runtime.local_rank,
        step_input_runtime.globalTid,
        step_input_runtime.pid,
        step_input_runtime.tid,
        step_input_runtime.function_name,
        step_input_runtime.event_name,
        CUPTI_ACTIVITY_KIND_MEMCPY.bytes,
        CUPTI_ACTIVITY_KIND_MEMCPY.correlationId,
        (CUPTI_ACTIVITY_KIND_MEMCPY.end - CUPTI_ACTIVITY_KIND_MEMCPY.start) AS duration,
        ((bytes * 1.0)/((CUPTI_ACTIVITY_KIND_MEMCPY.end - CUPTI_ACTIVITY_KIND_MEMCPY.start) * 0.000000001)) AS throughput 
    FROM step_input_runtime 
    JOIN CUPTI_ACTIVITY_KIND_MEMCPY 
        ON 
            step_input_runtime.pid == CUPTI_ACTIVITY_KIND_MEMCPY.globalPid / 0x1000000 % 0x1000000 AND
            step_input_runtime.correlationId == CUPTI_ACTIVITY_KIND_MEMCPY.correlationId;

DROP VIEW IF EXISTS step_communication_kernel;
CREATE VIEW step_communication_kernel AS 
    SELECT 
        step_communication_runtime.step_id,
        step_communication_runtime.rank,
        step_communication_runtime.local_rank,
        step_communication_runtime.globalTid,
        step_communication_runtime.event_name,
        step_communication_runtime.event_value,
        CUPTI_ACTIVITY_KIND_KERNEL.start,
        CUPTI_ACTIVITY_KIND_KERNEL.end, 
        (CUPTI_ACTIVITY_KIND_KERNEL.end - CUPTI_ACTIVITY_KIND_KERNEL.start) AS duration 
    FROM step_communication_runtime 
    JOIN CUPTI_ACTIVITY_KIND_KERNEL 
        ON 
            step_communication_runtime.pid == CUPTI_ACTIVITY_KIND_KERNEL.globalPid / 0x1000000 % 0x1000000 AND
            step_communication_runtime.correlationId == CUPTI_ACTIVITY_KIND_KERNEL.correlationId;

DROP VIEW IF EXISTS step_communication_transfer_kernel_start;
CREATE VIEW step_communication_transfer_kernel_start AS 
    SELECT 
        step_communication_kernel.step_id,
        step_communication_kernel.event_name,
        step_communication_kernel.event_value,
        MAX(step_communication_kernel.start) AS start
    FROM step_communication_kernel
    GROUP BY
        step_communication_kernel.step_id,
        step_communication_kernel.event_name,
        step_communication_kernel.event_value;

DROP VIEW IF EXISTS step_communication_wait_kernel;
CREATE VIEW step_communication_wait_kernel AS 
    SELECT 
        step_communication_kernel.step_id,
        step_communication_kernel.rank,
        step_communication_kernel.local_rank,
        step_communication_kernel.globalTid,
        step_communication_kernel.event_name,
        step_communication_kernel.event_value,
        IIF(step_communication_kernel.end <= step_communication_transfer_kernel_start.start, 0, step_communication_transfer_kernel_start.start - step_communication_kernel.start) AS duration 
    FROM step_communication_kernel
    JOIN step_communication_transfer_kernel_start
    ON
        step_communication_kernel.step_id == step_communication_transfer_kernel_start.step_id AND
        step_communication_kernel.event_name == step_communication_transfer_kernel_start.event_name AND
        step_communication_kernel.event_value == step_communication_transfer_kernel_start.event_value;

DROP VIEW IF EXISTS step_communication_transfer_kernel;
CREATE VIEW step_communication_transfer_kernel AS 
    SELECT 
        step_communication_kernel.step_id,
        step_communication_kernel.rank,
        step_communication_kernel.local_rank,
        step_communication_kernel.globalTid,
        step_communication_kernel.event_name,
        step_communication_kernel.event_value,
        IIF(step_communication_kernel.end <= step_communication_transfer_kernel_start.start, step_communication_kernel.end - step_communication_kernel.start, step_communication_kernel.end - step_communication_transfer_kernel_start.start) AS duration 
    FROM step_communication_kernel
    JOIN step_communication_transfer_kernel_start
    ON
        step_communication_kernel.step_id == step_communication_transfer_kernel_start.step_id AND
        step_communication_kernel.event_name == step_communication_transfer_kernel_start.event_name AND
        step_communication_kernel.event_value == step_communication_transfer_kernel_start.event_value;

DROP VIEW IF EXISTS step_compute_kernel;
CREATE VIEW step_compute_kernel AS 
    SELECT 
        step_compute_runtime.step_id,
        step_compute_runtime.rank,
        step_compute_runtime.local_rank,
        step_compute_runtime.globalTid,
        step_compute_runtime.event_name,
        step_compute_runtime.event_value,
        step_compute_runtime.function_name,
        (CUPTI_ACTIVITY_KIND_KERNEL.end - CUPTI_ACTIVITY_KIND_KERNEL.start) AS duration 
    FROM step_compute_runtime 
        JOIN CUPTI_ACTIVITY_KIND_KERNEL 
        ON 
            step_compute_runtime.pid == CUPTI_ACTIVITY_KIND_KERNEL.globalPid / 0x1000000 % 0x1000000 AND
            step_compute_runtime.correlationId == CUPTI_ACTIVITY_KIND_KERNEL.correlationId;


DROP VIEW IF EXISTS step_output_kernel;
CREATE VIEW step_output_kernel AS 
    SELECT 
        step_output_runtime.step_id,
        step_output_runtime.rank,
        step_output_runtime.local_rank,
        step_output_runtime.globalTid,
        step_output_runtime.event_name,
        CUPTI_ACTIVITY_KIND_MEMCPY.bytes,
        (CUPTI_ACTIVITY_KIND_MEMCPY.end - CUPTI_ACTIVITY_KIND_MEMCPY.start) AS duration,
        ((bytes * 1.0)/((CUPTI_ACTIVITY_KIND_MEMCPY.end - CUPTI_ACTIVITY_KIND_MEMCPY.start) * 0.000000001)) AS throughput 
    FROM step_output_runtime 
    JOIN CUPTI_ACTIVITY_KIND_MEMCPY 
    ON 
        step_output_runtime.pid == CUPTI_ACTIVITY_KIND_MEMCPY.globalPid / 0x1000000 % 0x1000000 AND
        step_output_runtime.correlationId == CUPTI_ACTIVITY_KIND_MEMCPY.correlationId;

DROP VIEW IF EXISTS step_input_host_stats;
CREATE VIEW step_input_host_stats AS
    SELECT
        step_input_range.step_id, 
        step_input_range.rank, 
        step_input_range.local_rank, 
        step_input_range.globalTid, 
        step_input_range.duration,
        step_input_range.event_name,
        sum(tensor_to_bytes(step_input_range.tensor)) AS bytes
    FROM step_input_range
    GROUP BY
        step_input_range.step_id,
        step_input_range.rank,
        step_input_range.local_rank;

DROP VIEW IF EXISTS step_output_host_stats;
CREATE VIEW step_output_host_stats AS
    SELECT
        step_output_range.step_id, 
        step_output_range.rank, 
        step_output_range.local_rank, 
        step_output_range.globalTid, 
        step_output_range.duration,
        step_output_range.event_name,
        sum(tensor_to_bytes(step_output_range.tensor)) AS bytes
    FROM step_output_range
    GROUP BY
        step_output_range.step_id,
        step_output_range.rank,
        step_output_range.local_rank,
        step_output_range.event_name;

DROP VIEW IF EXISTS step_communication_host_stats;
CREATE VIEW step_communication_host_stats AS
    SELECT
        step_communication_range.step_id, 
        step_communication_range.rank, 
        step_communication_range.local_rank, 
        step_communication_range.globalTid, 
        step_communication_range.duration,
        step_communication_range.event_name,
        step_communication_range.event_value,
        sum(tensors_to_bytes(step_communication_range.send_tensors)) AS send_bytes,
        sum(tensors_to_bytes(step_communication_range.recv_tensors)) AS recv_bytes
    FROM step_communication_range
    GROUP BY
        step_communication_range.step_id,
        step_communication_range.rank,
        step_communication_range.local_rank,
        step_communication_range.event_name,
        step_communication_range.event_value;

DROP VIEW IF EXISTS step_compute_host_stats;
CREATE VIEW step_compute_host_stats AS
    SELECT
        step_compute_range.step_id, 
        step_compute_range.rank, 
        step_compute_range.local_rank, 
        step_compute_range.globalTid, 
        step_compute_range.duration,
        step_compute_range.event_name,
        step_compute_range.event_value,
        sum(tensors_to_bytes(step_compute_range.weights)) AS weights 
    FROM step_compute_range
    GROUP BY
        step_compute_range.step_id,
        step_compute_range.rank,
        step_compute_range.local_rank,
        step_compute_range.event_name,
        step_compute_range.event_value;

DROP VIEW IF EXISTS step_input_device_stats;
CREATE VIEW step_input_device_stats AS 
    SELECT 
        step_input_kernel.step_id,
        step_input_kernel.rank,
        step_input_kernel.local_rank,
        step_input_kernel.globalTid,
        step_input_kernel.event_name,
        SUM(step_input_kernel.bytes) AS bytes, 
        SUM(step_input_kernel.duration) AS duration,
        AVG(step_input_kernel.throughput) AS throughput
    FROM step_input_kernel
    GROUP BY 
        step_input_kernel.step_id,
        step_input_kernel.rank,
        step_input_kernel.local_rank,
        step_input_kernel.event_name;

DROP VIEW IF EXISTS step_compute_device_stats;
CREATE VIEW step_compute_device_stats AS 
    SELECT 
        step_compute_kernel.step_id,
        step_compute_kernel.rank,
        step_compute_kernel.local_rank,
        step_compute_kernel.globalTid, 
        step_compute_kernel.event_name,
        step_compute_kernel.event_value,
        SUM(step_compute_kernel.duration) AS duration
    FROM step_compute_kernel
    GROUP BY 
        step_compute_kernel.step_id,
        step_compute_kernel.rank,
        step_compute_kernel.local_rank,
        step_compute_kernel.event_name,
        step_compute_kernel.event_value;

DROP VIEW IF EXISTS step_communication_wait_device_stats;
CREATE VIEW step_communication_wait_device_stats AS 
    SELECT 
        step_communication_wait_kernel.step_id,
        step_communication_wait_kernel.rank,
        step_communication_wait_kernel.local_rank,
        step_communication_wait_kernel.globalTid,
        step_communication_wait_kernel.event_name,
        step_communication_wait_kernel.event_value,
        SUM(step_communication_wait_kernel.duration) AS duration
    FROM step_communication_wait_kernel 
    GROUP BY 
        step_communication_wait_kernel.step_id,
        step_communication_wait_kernel.rank,
        step_communication_wait_kernel.local_rank,
        step_communication_wait_kernel.event_name,
        step_communication_wait_kernel.event_value;

DROP VIEW IF EXISTS step_communication_transfer_device_stats;
CREATE VIEW step_communication_transfer_device_stats AS 
    SELECT 
        step_communication_transfer_kernel.step_id,
        step_communication_transfer_kernel.rank,
        step_communication_transfer_kernel.local_rank,
        step_communication_transfer_kernel.globalTid,
        step_communication_transfer_kernel.event_name,
        step_communication_transfer_kernel.event_value,
        SUM(step_communication_transfer_kernel.duration) AS duration
    FROM step_communication_transfer_kernel 
    GROUP BY 
        step_communication_transfer_kernel.step_id,
        step_communication_transfer_kernel.rank,
        step_communication_transfer_kernel.local_rank,
        step_communication_transfer_kernel.event_name,
        step_communication_transfer_kernel.event_value;

DROP VIEW IF EXISTS step_output_device_stats;
CREATE VIEW step_output_device_stats AS 
    SELECT 
        step_output_kernel.step_id,
        step_output_kernel.rank,
        step_output_kernel.local_rank,
        step_output_kernel.globalTid,
        step_output_kernel.event_name,
        SUM(step_output_kernel.bytes) AS bytes,
        SUM(step_output_kernel.duration) AS duration,
        AVG(step_output_kernel.throughput) AS throughput
    FROM step_output_kernel 
    GROUP BY 
        step_output_kernel.step_id,
        step_output_kernel.rank,
        step_output_kernel.local_rank,
        step_output_kernel.event_name;

DROP VIEW IF EXISTS step_input_stats;
CREATE VIEW step_input_stats AS 
    SELECT 
        step_input_host_stats.step_id, 
        step_input_host_stats.rank,
        step_input_host_stats.local_rank,
        step_input_host_stats.globalTid,
        step_input_host_stats.duration AS host_duration, 
        step_input_host_stats.bytes AS host_bytes, 
        step_input_device_stats.duration AS device_duration,
        step_input_device_stats.bytes AS device_bytes,
        step_input_device_stats.throughput
    FROM step_input_host_stats
    JOIN step_input_device_stats
    ON 
        step_input_host_stats.step_id == step_input_device_stats.step_id AND
        step_input_host_stats.rank == step_input_device_stats.rank AND
        step_input_host_stats.local_rank == step_input_device_stats.local_rank;

DROP VIEW IF EXISTS step_compute_stats;
CREATE VIEW step_compute_stats AS 
    SELECT 
        step_compute_host_stats.step_id,
        step_compute_host_stats.rank,
        step_compute_host_stats.local_rank,
        step_compute_host_stats.globalTid,
        step_compute_host_stats.event_name,
        step_compute_host_stats.event_value,
        step_compute_host_stats.weights,
        step_compute_host_stats.duration AS host_duration,
        step_compute_device_stats.duration AS device_duration
    FROM step_compute_host_stats
    JOIN step_compute_device_stats
    ON 
        step_compute_host_stats.step_id == step_compute_device_stats.step_id AND
        step_compute_host_stats.rank == step_compute_device_stats.rank AND
        step_compute_host_stats.local_rank == step_compute_device_stats.local_rank AND
        step_compute_host_stats.event_name == step_compute_device_stats.event_name AND
        step_compute_host_stats.event_value == step_compute_device_stats.event_value;

DROP VIEW IF EXISTS step_communication_wait_stats;
CREATE VIEW step_communication_wait_stats AS 
    SELECT 
        step_communication_host_stats.step_id,
        step_communication_host_stats.rank,
        step_communication_host_stats.local_rank,
        step_communication_host_stats.globalTid,
        step_communication_host_stats.duration AS host_duration,
        step_communication_host_stats.send_bytes,
        step_communication_host_stats.recv_bytes,
        step_communication_wait_device_stats.duration AS device_duration
    FROM step_communication_host_stats
    JOIN step_communication_wait_device_stats
    ON 
        step_communication_host_stats.step_id == step_communication_wait_device_stats.step_id AND
        step_communication_host_stats.rank == step_communication_wait_device_stats.rank AND
        step_communication_host_stats.local_rank == step_communication_wait_device_stats.local_rank AND
        step_communication_host_stats.event_name == step_communication_wait_device_stats.event_name AND
        step_communication_host_stats.event_value == step_communication_wait_device_stats.event_value;

DROP VIEW IF EXISTS step_communication_transfer_stats;
CREATE VIEW step_communication_transfer_stats AS 
    SELECT 
        step_communication_host_stats.step_id,
        step_communication_host_stats.rank,
        step_communication_host_stats.local_rank,
        step_communication_host_stats.globalTid,
        step_communication_host_stats.duration AS host_duration,
        step_communication_host_stats.send_bytes,
        step_communication_host_stats.recv_bytes,
        step_communication_transfer_device_stats.duration AS device_duration
    FROM step_communication_host_stats
    JOIN step_communication_transfer_device_stats
    ON 
        step_communication_host_stats.step_id == step_communication_transfer_device_stats.step_id AND
        step_communication_host_stats.rank == step_communication_transfer_device_stats.rank AND
        step_communication_host_stats.local_rank == step_communication_transfer_device_stats.local_rank AND
        step_communication_host_stats.event_name == step_communication_transfer_device_stats.event_name AND
        step_communication_host_stats.event_value == step_communication_transfer_device_stats.event_value;

DROP VIEW IF EXISTS step_output_stats;
CREATE VIEW step_output_stats AS 
    SELECT 
        step_output_host_stats.step_id, 
        step_output_host_stats.rank,
        step_output_host_stats.local_rank,
        step_output_host_stats.globalTid,
        step_output_host_stats.bytes As host_bytes, 
        step_output_host_stats.duration AS host_duration, 
        step_output_device_stats.duration AS device_duration,
        step_output_device_stats.bytes As device_bytes, 
        step_output_device_stats.throughput
    FROM step_output_host_stats 
    JOIN step_output_device_stats
    ON 
        step_output_host_stats.step_id == step_output_device_stats.step_id AND
        step_output_host_stats.rank == step_output_device_stats.rank AND
        step_output_host_stats.local_rank == step_output_device_stats.local_rank;

DROP VIEW IF EXISTS step_inference_stats;
CREATE VIEW step_inference_stats AS 
    SELECT 
        step_inference_range.step_id, 
        step_inference_range.batch_size, 
        step_inference_range.rank,
        step_inference_range.local_rank,
        step_inference_range.globalTid,
        step_inference_range.start AS host_start,
        step_inference_range.end AS host_end,
        step_inference_range.duration AS host_duration
    FROM step_inference_range