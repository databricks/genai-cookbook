#!/bin/bash
fswatch -o --exclude "../genai_cookbook/_build" ../genai_cookbook | xargs -n1 -I{} jupyter-book build --all ../genai_cookbook