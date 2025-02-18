# candle-examples-moondream

following [github.com/huggingface/candle/tree/main/candle-examples/examples/moondream](https://github.com/huggingface/candle/tree/main/candle-examples/examples/moondream)

## usage

safetensors

```bash
$ cargo run --release -- --prompt "What is in the picture?" --image "./demo-1.jpg"
```

quantized GGUF

```bash
$ cargo run --release -- --prompt "What is in the picture?" --image "./demo-1.jpg" --quantized
```
