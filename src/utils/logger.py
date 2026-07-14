# src/utils/logger.py
"""
Terminal logger used by every pipeline stage.
Outputs coloured, timestamped, structured messages so you can
follow exactly what each DVC stage is doing in real time.

Usage (in any stage script):
    from src.utils.logger import get_logger
    logger = get_logger("data_ingestion")
    logger.info("Loading raw CSV ...")
    logger.success("Saved 1280 rows → data/processed/train.csv")
    logger.step("STAGE 1 COMPLETE")
"""

import logging
import sys
from datetime import datetime

# ── ANSI colour codes ────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
GREY    = "\033[90m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
MAGENTA = "\033[95m"
BLUE    = "\033[94m"
WHITE   = "\033[97m"

# ── Custom log levels ────────────────────────────────────────────────
SUCCESS_LEVEL = 25          # between INFO(20) and WARNING(30)
STEP_LEVEL    = 35          # between WARNING(30) and ERROR(40)

logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")
logging.addLevelName(STEP_LEVEL,    "STEP")


class _ColouredFormatter(logging.Formatter):
    """Maps each log level to a distinct colour + symbol."""

    LEVEL_STYLES = {
        logging.DEBUG:   (GREY,    "·"),
        logging.INFO:    (CYAN,    "ℹ"),
        SUCCESS_LEVEL:   (GREEN,   "✔"),
        logging.WARNING: (YELLOW,  "⚠"),
        STEP_LEVEL:      (MAGENTA, "▶"),
        logging.ERROR:   (RED,     "✖"),
        logging.CRITICAL:(RED,     "☠"),
    }

    def format(self, record: logging.LogRecord) -> str:
        colour, symbol = self.LEVEL_STYLES.get(record.levelno, (WHITE, "?"))
        ts    = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        level = f"{colour}{BOLD}{symbol} {record.levelname:<7}{RESET}"
        name  = f"{GREY}[{record.name}]{RESET}"
        msg   = f"{colour}{record.getMessage()}{RESET}"
        return f"{GREY}{ts}{RESET}  {level}  {name}  {msg}"


class _PipelineLogger(logging.Logger):
    """Logger subclass that adds .success() and .step() methods."""

    def success(self, msg: str, *args, **kwargs):
        if self.isEnabledFor(SUCCESS_LEVEL):
            self._log(SUCCESS_LEVEL, msg, args, **kwargs)

    def step(self, msg: str, *args, **kwargs):
        """Use for major stage boundaries — prints a banner line."""
        if self.isEnabledFor(STEP_LEVEL):
            banner = f"\n{'═'*60}\n  {msg}\n{'═'*60}"
            self._log(STEP_LEVEL, banner, args, **kwargs)


logging.setLoggerClass(_PipelineLogger)


def get_logger(name: str, level: int = logging.DEBUG) -> _PipelineLogger:
    """
    Return a configured _PipelineLogger for the given stage name.

    Parameters
    ----------
    name  : str   — typically the script/stage name, e.g. "stage_01_data_ingestion"
    level : int   — minimum log level (default DEBUG → shows everything)
    """
    logger: _PipelineLogger = logging.getLogger(name)   # type: ignore[assignment]

    if logger.handlers:
        # already configured (happens if a module is imported twice)
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(_ColouredFormatter())
    logger.addHandler(handler)
    logger.propagate = False

    return logger