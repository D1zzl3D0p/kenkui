import logging
import threading
from collections.abc import Callable

from .chapter_filter import FilterOperation

logger = logging.getLogger(__name__)
from .models import AppConfig, ChapterPreset, JobConfig, ProcessingConfig, QueueItem
from .parsing import AudioBuilder


class Processor:
    def __init__(self):
        self._thread: threading.Thread | None = None
        self._running = False
        self._current_item: QueueItem | None = None
        self._callbacks: dict[str, Callable] = {}
        self._progress_callback: Callable[[float, str, int], None] | None = None

    def on(self, event: str, callback: Callable):
        self._callbacks[event] = callback

    def _emit(self, event: str, *args):
        if event in self._callbacks:
            self._callbacks[event](*args)

    def start(
        self,
        item: QueueItem,
        app_config: AppConfig,
        progress_callback: Callable[[float, str, int], None] | None = None,
    ):
        logger.debug(f"start() called with item {item.id}")
        logger.debug(f"_running before: {self._running}")
        if self._running:
            logger.debug("[DEBUG Processor] Already running - returning False")
            return False
        self._current_item = item
        self._running = True
        logger.debug("[DEBUG Processor] _running set to True")
        self._progress_callback = progress_callback
        logger.debug("[DEBUG Processor] Creating thread...")
        self._thread = threading.Thread(target=self._process, args=(item, app_config))
        self._thread.start()
        logger.debug(f"Thread started: {self._thread.name}")
        return True

    def stop(self):
        logger.debug("[DEBUG Processor] stop() called")
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _process(self, item: QueueItem, app_config: AppConfig):
        logger.debug(f"_process() running for item {item.id}")
        job = item.job
        logger.debug("[DEBUG Processor] About to emit 'started' event")
        self._emit("started", item.id)
        logger.debug("[DEBUG Processor] Emitted 'started' event")

        try:
            logger.debug("[DEBUG Processor] Building config...")
            cfg = self._build_config(job, app_config)
            logger.debug("[DEBUG Processor] Creating AudioBuilder...")
            builder = AudioBuilder(cfg, progress_callback=self._progress_callback)
            logger.debug("[DEBUG Processor] AudioBuilder created, about to call run()")

            self._emit("progress", item.id, 0.0, "Starting...", 0)
            logger.debug("Emitted progress event - calling builder.run()")

            result = builder.run()
            logger.debug(f"builder.run() returned: {result}")

            if result:
                self._emit("completed", item.id)
                logger.debug("[DEBUG Processor] Emitted 'completed' event")
            else:
                self._emit("failed", item.id, "Conversion failed")
                logger.debug("[DEBUG Processor] Emitted 'failed' event")

        except Exception as e:
            logger.debug(f"Exception in _process: {e}")
            import traceback

            logger.debug(f"Traceback: {traceback.format_exc()}")
            self._emit("failed", item.id, str(e))
        finally:
            logger.debug("[DEBUG Processor] Setting _running to False in finally block")
            self._running = False
            self._current_item = None

    def _build_config(self, job: JobConfig, app_config: AppConfig) -> ProcessingConfig:
        preset = job.chapter_selection.preset
        if preset in (ChapterPreset.MANUAL, ChapterPreset.CUSTOM):
            # Use explicit index list from the UI checkbox selection
            operations = [
                FilterOperation("index", str(idx))
                for idx in job.chapter_selection.included
            ]
            if not operations:
                operations = [FilterOperation("preset", "content-only")]
        else:
            operations = [FilterOperation("preset", preset.value)]

        output_path = job.output_path or job.ebook_path.parent

        return ProcessingConfig(
            voice=job.voice,
            ebook_path=job.ebook_path,
            output_path=output_path,
            pause_line_ms=app_config.pause_line_ms,
            pause_chapter_ms=app_config.pause_chapter_ms,
            workers=app_config.workers,
            m4b_bitrate=app_config.m4b_bitrate,
            keep_temp=app_config.keep_temp,
            debug_html=app_config.verbose,
            chapter_filters=operations,
            verbose=app_config.verbose,
            temp=app_config.temp,
            lsd_decode_steps=app_config.lsd_decode_steps,
            noise_clamp=app_config.noise_clamp,
        )

    @property
    def is_running(self) -> bool:
        return self._running

    def reset(self):
        """Reset processor state - use when stuck."""
        self._running = False
        self._current_item = None
        self._thread = None


_processor: Processor | None = None


def reset_processor():
    """Reset the global processor instance."""
    global _processor
    if _processor:
        _processor.reset()
    _processor = None


def get_processor() -> Processor:
    global _processor
    if _processor is None:
        _processor = Processor()
    return _processor
