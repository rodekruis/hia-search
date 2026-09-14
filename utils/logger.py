"""Application logger.

Handlers and exporters are configured by utils.telemetry.configure_telemetry()
at startup; importing this module has no side effects.
"""

import logging

logger = logging.getLogger("hia-search")
