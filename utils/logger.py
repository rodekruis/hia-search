import logging
import os
import sys
from dotenv import load_dotenv
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from azure.monitor.opentelemetry.exporter import AzureMonitorLogExporter

# load environment variables
load_dotenv()

# Set up logs export to Azure Application Insights
logger_provider = LoggerProvider()
set_logger_provider(logger_provider)
exporter = AzureMonitorLogExporter(
    connection_string=os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"]
)
logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporter))

# Attach OpenTelemetry handler to root logger
otel_handler = LoggingHandler()
logging.getLogger().addHandler(otel_handler)

# Attach console handler for stdout output
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s : %(levelname)s : %(message)s")
)
logging.getLogger().addHandler(console_handler)

logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# Silence noisy loggers
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("requests_oauthlib").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("opentelemetry").setLevel(logging.ERROR)
