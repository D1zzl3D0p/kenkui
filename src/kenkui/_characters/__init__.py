"""Character inference, dialogue attribution, and their durable store.

The only part of Kenkui that calls a language model. Everything here runs in
the parent process during resolution, never inside a spawned render worker, so
the render path's offline posture is untouched.
"""
