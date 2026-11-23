# Setup OpenTelemetry POC

Trace export uses the retained frontend and `vllm` instrumentation names.
The examples below configure a target-model endpoint; service names are local
labels. The upstream screenshots and `dummy_client.py` are references.
Use a compatible collector and validate exported spans in your deployment.

> **Note:** The core OpenTelemetry packages (`opentelemetry-sdk`, `opentelemetry-api`, `opentelemetry-exporter-otlp`, `opentelemetry-semantic-conventions-ai`) are bundled with vLLM. Manual installation is not required.

1. Start Jaeger in a docker container:

    ```bash
    # From: https://www.jaegertracing.io/docs/1.57/getting-started/
    docker run --rm --name jaeger \
        -e COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
        -p 6831:6831/udp \
        -p 6832:6832/udp \
        -p 5778:5778 \
        -p 16686:16686 \
        -p 4317:4317 \
        -p 4318:4318 \
        -p 14250:14250 \
        -p 14268:14268 \
        -p 14269:14269 \
        -p 9411:9411 \
        jaegertracing/all-in-one:1.57
    ```

1. In a new shell, export Jaeger IP:

    ```bash
    export JAEGER_IP=$(docker inspect   --format '{{ .NetworkSettings.IPAddress }}' jaeger)
    export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=grpc://$JAEGER_IP:4317
    ```

    Then set vLLM's service name for OpenTelemetry, enable insecure connections to Jaeger and run vLLM:

    ```bash
    export OTEL_SERVICE_NAME="nabla-server"
    export OTEL_EXPORTER_OTLP_TRACES_INSECURE=true
    nabla-vllm serve /models/Qwen3.8-Flash-Next-FP8 \
      --served-model-name nabla-model \
      --tensor-parallel-size 4 --enable-expert-parallel --language-model-only \
      --engram-config '{"cpu_offload":true}' --otlp-traces-endpoint="$OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
    ```

1. In a new shell, send a request with explicit trace context

    ```bash
    export JAEGER_IP=$(docker inspect --format '{{ .NetworkSettings.IPAddress }}' jaeger)
    export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=grpc://$JAEGER_IP:4317
    export OTEL_EXPORTER_OTLP_TRACES_INSECURE=true
    export OTEL_SERVICE_NAME="client-service"
    curl http://localhost:8000/v1/completions \
      -H 'Content-Type: application/json' \
      -H 'traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01' \
      -d '{"model":"nabla-model","prompt":"Explain batching.","max_tokens":10}'
    ```

1. Open Jaeger webui: <http://localhost:16686/>

    In the search pane, select `nabla-server` service and hit `Find Traces`. You should get a list of traces, one for each request.
    ![Traces](https://i.imgur.com/GYHhFjo.png)

1. Clicking on a trace will show its spans and their tags. The spans depend on the configured instrumentation. The screenshots below
show the inherited upstream dummy-client example; they are not captures of
the curl request above.
![Spans details](https://i.imgur.com/OPf6CBL.png)

## Exporter Protocol

OpenTelemetry supports either `grpc` or `http/protobuf` as the transport protocol for trace data in the exporter.
By default, `grpc` is used. To set `http/protobuf` as the protocol, configure the `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` environment variable as follows:

```bash
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://$JAEGER_IP:4318/v1/traces
nabla-vllm serve /models/Qwen3.8-Flash-Next-FP8 \
      --served-model-name nabla-model \
      --tensor-parallel-size 4 --enable-expert-parallel --language-model-only \
      --engram-config '{"cpu_offload":true}' --otlp-traces-endpoint="$OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
```

## Instrumentation of FastAPI

OpenTelemetry allows automatic instrumentation of FastAPI.

1. Install the instrumentation library

    ```bash
    uv pip install opentelemetry-instrumentation-fastapi
    ```

1. Run vLLM with `opentelemetry-instrument`

    ```bash
    opentelemetry-instrument nabla-vllm serve /models/Qwen3.8-Flash-Next-FP8 \
      --served-model-name nabla-model \
      --tensor-parallel-size 4 --enable-expert-parallel --language-model-only \
      --engram-config '{"cpu_offload":true}'
    ```

1. Send a request to vLLM and find its trace in Jaeger. It should contain spans from FastAPI.

![FastAPI Spans](https://i.imgur.com/hywvoOJ.png)
