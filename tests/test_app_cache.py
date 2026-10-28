"""Browser caching: index.html is revalidated every load, hashed /assets are cached forever."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
app_module = pytest.importorskip("app")
if not app_module.DIST_DIR.exists():
    pytest.skip("frontend/dist is not built", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

# no `with`: the lifespan (model loading) never runs
client = TestClient(app_module.app)


@pytest.mark.parametrize("path", ["/", "/index.html", "/some/client/route"])
def test_index_html_is_revalidated(path):
    r = client.get(path)
    assert r.status_code == 200
    assert r.headers["cache-control"] == "no-cache"
    assert "<html" in r.text.lower()


def test_hashed_assets_stay_immutable():
    asset = next((app_module.DIST_DIR / "assets").glob("index-*.js"))
    r = client.get(f"/assets/{asset.name}")
    assert r.status_code == 200
    assert "immutable" in r.headers["cache-control"]


def test_other_dist_files_are_not_marked_no_cache():
    r = client.get("/samples/manifest.json")
    assert r.status_code == 200
    assert "no-cache" not in r.headers.get("cache-control", "")
