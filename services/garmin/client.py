import logging
import os
from collections.abc import Callable
from pathlib import Path

from garminconnect import Garmin

logger = logging.getLogger(__name__)


class GarminConnectClient:
    def __init__(self, token_dir: str | None = None):
        self._client: Garmin | None = None
        self._token_dir = Path(
            token_dir
            or os.getenv("GARMINCONNECT_TOKENS")
            or os.path.expanduser("~/.garminconnect")
        )

    def connect(
        self,
        email: str,
        password: str,
        mfa_callback: Callable[[], str] | None = None,
    ) -> None:
        logger.info("Initializing Garmin Connect client")
        self._token_dir.mkdir(parents=True, exist_ok=True)

        prompt_mfa = mfa_callback or (lambda: input("MFA code: "))
        self._client = Garmin(
            email=email,
            password=password,
            prompt_mfa=prompt_mfa,
        )
        self._client.login(tokenstore=str(self._token_dir))
        logger.info("Successfully connected to Garmin Connect")

    @property
    def client(self) -> Garmin:
        if self._client is None:
            raise RuntimeError("GarminConnectClient not connected")
        return self._client

    def get_heat_altitude_acclimatization(self, start_date: str, end_date: str) -> dict | None:
        if not self._client:
            logger.error("Cannot fetch heat/altitude acclimatization: client not connected")
            return None

        try:
            url = f"/metrics-service/metrics/heataltitudeacclimation/daily/{start_date}/{end_date}"
            logger.info(
                "Fetching heat/altitude acclimatization from %s to %s", start_date, end_date
            )
            data = self._client.connectapi(url)
            return data
        except Exception as exc:
            logger.exception("Failed to fetch heat/altitude acclimatization data: %s", exc)
            return None

    def disconnect(self) -> None:
        if self._client:
            self._client = None
            logger.info("Disconnected from Garmin Connect")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, _exc_val, _exc_tb):
        self.disconnect()
