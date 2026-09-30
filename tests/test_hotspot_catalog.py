import os
import re

from model.live_satellite import LIVE_HOTSPOTS, WAYBACK_RELEASES

INDEX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates", "index.html")


def _html():
    with open(INDEX, encoding="utf-8") as f:
        return f.read()


def test_hotspot_years_have_wayback_releases():
    for h in LIVE_HOTSPOTS.values():
        assert h["year_t1"] in WAYBACK_RELEASES and h["year_t2"] in WAYBACK_RELEASES


def test_hotspot_select_lists_exactly_the_live_hotspots():
    html = _html()
    select = re.search(r'<select id="select-hotspot"[^>]*>(.*?)</select>', html, re.S).group(1)
    assert re.findall(r'<option value="([^"]+)"', select) == list(LIVE_HOTSPOTS)


def test_discover_cards_match_hotspot_ids_and_years():
    """The card's year label must be what the analysis actually compares."""
    cards = re.findall(r'<div class="discover-card" data-hotspot="([^"]+)">(.*?)</button>', _html(), re.S)
    assert [hid for hid, _ in cards] == list(LIVE_HOTSPOTS)
    for hid, body in cards:
        years = re.search(r'</i>\s*(\d{4}) → (\d{4})</span>', body).groups()
        assert years == (LIVE_HOTSPOTS[hid]["year_t1"], LIVE_HOTSPOTS[hid]["year_t2"]), hid
