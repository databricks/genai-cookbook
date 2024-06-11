from livereload import Server, shell
import os

# Define the subfolder you want to serve
subfolder = '../genai_cookbook/_build/html/'

# Create a new server instance
server = Server()

# Watch the subfolder (and subdirectories) for changes to HTML, CSS, and JavaScript files
server.watch(subfolder, shell('touch reload.txt'), delay=1)

# Serve the specified subfolder on port 8000
server.serve(root=subfolder, port=8000)