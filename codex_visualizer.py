#!/usr/bin/env python3
"""Create a detailed Kaggriculture action report from replay.json."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import webbrowser


def escape(value: object) -> str:
    return html.escape(str(value))


def format_money(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "Unknown"

    if float(value).is_integer():
        return f"${value:,.0f}"

    return f"${value:,.2f}"


def format_position(position: object) -> str:
    if isinstance(position, list) and len(position) >= 2:
        return f"({escape(position[0])}, {escape(position[1])})"

    return "(?, ?)"


def format_action(action: object) -> str:
    if not isinstance(action, list) or not action:
        value = html.escape(json.dumps(action, ensure_ascii=False))
        return f'<span class="action unknown">{value}</span>'

    operation = escape(action[0])
    arguments = " ".join(escape(value) for value in action[1:])
    css_class = " pass" if action == ["PASS"] else ""

    arguments_html = (
        f'<span class="arguments">{arguments}</span>'
        if arguments
        else ""
    )

    return (
        f'<span class="action{css_class}">'
        f"<strong>{operation}</strong>"
        f"{arguments_html}"
        "</span>"
    )


def format_hand_actions(
    actions: object,
    positions: object,
) -> str:
    if not isinstance(actions, list):
        actions = []

    if not isinstance(positions, list):
        positions = []

    # Prefer the number of actual hired hands. Some agents submit
    # extra PASS placeholders for hands that do not exist.
    if positions:
        hand_count = len(positions)
    else:
        hand_count = len(actions)

    if hand_count == 0:
        return '<span class="empty">No hired hands</span>'

    lines = []

    for index in range(hand_count):
        action = (
            actions[index]
            if index < len(actions)
            else ["PASS"]
        )

        position = (
            positions[index]
            if index < len(positions)
            else None
        )

        lines.append(
            '<div class="hand-card">'
            '<div class="hand-heading">'
            f"Hand {index + 1} {format_position(position)}"
            "</div>"
            '<div class="hand-action">'
            f"{format_action(action)}"
            "</div>"
            "</div>"
        )

    return "".join(lines)


def format_market_actions(
    actions: object,
    market_prices: dict,
) -> str:
    if not isinstance(actions, list) or not actions:
        return '<span class="empty">No market actions</span>'

    operation_labels = {
        "BUY_SEED": "Buy seed",
        "BUY_PRODUCT": "Buy product",
        "BUY_ANIMAL": "Buy animal",
        "SELL": "Sell",
        "BUY_LAND": "Buy land",
        "HIRE": "Hire hand",
    }

    cards = []

    for action in actions:
        if not isinstance(action, list) or not action:
            continue

        operation = str(action[0])
        item = action[1] if len(action) >= 2 else None
        quantity = action[2] if len(action) >= 3 else 1

        operation_label = operation_labels.get(
            operation,
            operation.replace("_", " ").title(),
        )

        if operation.startswith("BUY"):
            order_class = "buy"
        elif operation == "SELL":
            order_class = "sell"
        else:
            order_class = "other"

        if item is not None:
            item_name = escape(item).replace("_", " ").title()

            description_html = (
                '<div class="order-description">'
                f'<strong class="order-item">{item_name}</strong>'
                f'<span class="order-quantity">× {escape(quantity)}</span>'
                "</div>"
            )
        else:
            description_html = ""

        unit_price = market_prices.get(item)
        pricing_html = ""

        if isinstance(unit_price, (int, float)):
            numeric_quantity = (
                quantity
                if isinstance(quantity, (int, float))
                else 1
            )

            total_price = unit_price * numeric_quantity

            pricing_html = (
                '<div class="order-pricing">'
                '<span class="unit-price">'
                f"{format_money(unit_price)} each"
                "</span>"
                '<strong class="total-price">'
                f"{format_money(total_price)} total"
                "</strong>"
                "</div>"
            )
        elif operation in {
            "BUY_SEED",
            "BUY_PRODUCT",
            "BUY_ANIMAL",
            "SELL",
        }:
            pricing_html = (
                '<div class="order-pricing">'
                '<span class="price-unavailable">'
                "Price unavailable"
                "</span>"
                "</div>"
            )

        cards.append(
            f'<div class="market-order {order_class}">'
            '<div class="order-heading">'
            f'<span class="order-type">{escape(operation_label)}</span>'
            f"{description_html}"
            "</div>"
            f"{pricing_html}"
            "</div>"
        )

    return "".join(cards)


def is_idle(player: object) -> bool:
    if not isinstance(player, dict):
        return True

    action = player.get("action", {})

    if not isinstance(action, dict):
        return True

    farmer_idle = action.get("farmer", ["PASS"]) == ["PASS"]

    hands = action.get("hands", [])
    hands_idle = (
        isinstance(hands, list)
        and all(hand_action == ["PASS"] for hand_action in hands)
    )

    market_idle = action.get("market", []) == []

    return farmer_idle and hands_idle and market_idle


def find_shared_observation(step: list) -> dict:
    for player in step:
        if not isinstance(player, dict):
            continue

        observation = player.get("observation", {})

        if (
            isinstance(observation, dict)
            and (
                "farms" in observation
                or "market" in observation
            )
        ):
            return observation

    return {}


def market_prices_from_observation(
    observation: dict,
) -> dict:
    market = observation.get("market", {})

    if not isinstance(market, dict):
        return {}

    prices = market.get("prices", {})

    return prices if isinstance(prices, dict) else {}


def get_farm_state(
    observation: dict,
    player_number: int,
) -> dict:
    farms = observation.get("farms", [])

    if (
        isinstance(farms, list)
        and player_number < len(farms)
        and isinstance(farms[player_number], dict)
    ):
        return farms[player_number]

    return {}


def player_cells(
    player: object,
    farm: dict,
    transaction_prices: dict,
) -> str:
    if not isinstance(player, dict):
        player = {}

    action = player.get("action", {})

    if not isinstance(action, dict):
        action = {}

    money = format_money(farm.get("money"))
    farmer_position = format_position(farm.get("farmer"))
    hand_positions = farm.get("hands", [])

    farmer_action = format_action(
        action.get("farmer", ["PASS"])
    )

    hand_actions = format_hand_actions(
        action.get("hands", []),
        hand_positions,
    )

    market_actions = format_market_actions(
        action.get("market", []),
        transaction_prices,
    )

    return (
        '<td class="farmer-cell">'

        '<div class="balance-card">'
        '<span class="balance-label">Current money</span>'
        f'<strong class="balance-amount">{money}</strong>'
        "</div>"

        '<div class="farmer-position">'
        '<span>Farmer position</span>'
        f"<strong>{farmer_position}</strong>"
        "</div>"

        '<div class="section-label">Farmer action</div>'
        f"{farmer_action}"

        "</td>"

        '<td class="hands-cell">'
        f"{hand_actions}"
        "</td>"

        '<td class="market-cell">'
        f"{market_actions}"
        "</td>"
    )


def step_time_html(
    step_number: int,
    observation: dict,
) -> str:
    day = observation.get("day")
    hour = observation.get("hour")

    if isinstance(day, int):
        # Replay days are zero-indexed.
        day += 1

    day_text = "?" if day is None else escape(day)
    hour_text = "?" if hour is None else escape(hour)

    return (
        f'<strong class="step-number">Step {step_number}</strong>'
        f'<span class="day">Day {day_text}</span>'
        f'<span class="hour">Hour {hour_text}</span>'
    )


def build_actions_html(
    replay: dict,
    source_name: str,
) -> str:
    steps = replay.get("steps")

    if not isinstance(steps, list):
        raise ValueError(
            "Replay JSON must contain a top-level 'steps' list."
        )

    observations = [
        find_shared_observation(step)
        if isinstance(step, list)
        else {}
        for step in steps
    ]

    rows = []

    for step_number, step in enumerate(steps):
        if not isinstance(step, list):
            raise ValueError(
                f"Step {step_number} is not a list of player records."
            )

        players = list(step) + [{}, {}]
        player_zero = players[0]
        player_one = players[1]

        observation = observations[step_number]

        current_prices = market_prices_from_observation(
            observation
        )

        # Actions on step N were selected using the state from
        # step N-1. Use that market snapshot for transaction prices.
        if step_number > 0:
            transaction_prices = (
                market_prices_from_observation(
                    observations[step_number - 1]
                )
            )
        else:
            transaction_prices = current_prices

        player_zero_farm = get_farm_state(
            observation,
            0,
        )

        player_one_farm = get_farm_state(
            observation,
            1,
        )

        idle = (
            is_idle(player_zero)
            and is_idle(player_one)
        )

        market_data = html.escape(
            json.dumps(
                current_prices,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            quote=True,
        )

        rows.append(
            f'<tr id="step-{step_number}" '
            f'data-step="{step_number}" '
            f'data-idle="{str(idle).lower()}" '
            f'data-market="{market_data}">'

            '<td class="step-time">'
            f"{step_time_html(step_number, observation)}"
            "</td>"

            f"{player_cells(
                player_zero,
                player_zero_farm,
                transaction_prices,
            )}"

            f"{player_cells(
                player_one,
                player_one_farm,
                transaction_prices,
            )}"

            "</tr>"
        )

    title = html.escape(
        str(
            replay.get("title")
            or replay.get("name")
            or "Replay"
        )
    )

    source = html.escape(source_name)

    rewards = html.escape(
        json.dumps(
            replay.get("rewards", []),
            ensure_ascii=False,
        )
    )

    statuses = html.escape(
        json.dumps(
            replay.get("statuses", []),
            ensure_ascii=False,
        )
    )

    last_step = max(len(steps) - 1, 0)

    page = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>__TITLE__ action report</title>

<style>
:root {
    color-scheme: dark;
    --background: #0d1117;
    --panel: #161b22;
    --header: #21262d;
    --border: #30363d;
    --muted: #8b949e;
    --text: #e6edf3;
    --blue: #58a6ff;
    --green: #7ee787;
    --purple: #d2a8ff;
    --gold: #f2cc60;
    --orange: #f0883e;
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: var(--background);
    color: var(--text);
    font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
}

header {
    position: relative;
    min-height: 145px;
    padding: 24px 370px 18px 32px;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(135deg, #161b22, #111827);
}

h1 {
    margin: 0 0 8px;
    font-size: clamp(24px, 3vw, 34px);
}

.meta {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 20px;
    color: var(--muted);
}

.meta strong {
    color: var(--text);
}

.market-panel {
    position: absolute;
    top: 16px;
    right: 24px;
    width: 320px;
    max-height: 115px;
    padding: 11px 13px;
    overflow: auto;
    border: 1px solid var(--border);
    border-radius: 9px;
    background: rgba(13, 17, 23, 0.95);
}

.market-panel-header {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 7px;
}

.market-panel-header strong {
    color: white;
}

#market-step {
    color: var(--muted);
    font-size: 12px;
}

.market-prices {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 4px 10px;
}

.market-price {
    display: flex;
    justify-content: space-between;
    gap: 6px;
    color: var(--muted);
    font-size: 11px;
}

.market-price strong {
    color: var(--gold);
    font-weight: 650;
}

.toolbar {
    display: flex;
    flex-wrap: wrap;
    align-items: end;
    gap: 10px;
    padding: 12px 32px;
    border-bottom: 1px solid var(--border);
    background: rgba(13, 17, 23, 0.96);
}

.control {
    display: grid;
    gap: 4px;
    color: var(--muted);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

input,
button {
    min-height: 36px;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: var(--panel);
    color: var(--text);
    font: inherit;
}

input {
    width: 220px;
    padding: 7px 10px;
}

input[type="number"] {
    width: 100px;
}

input:focus {
    border-color: var(--blue);
    outline: none;
    box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.15);
}

button {
    padding: 7px 14px;
    cursor: pointer;
}

button:hover {
    border-color: #6e7681;
    background: #21262d;
}

.checkbox-control {
    display: flex;
    align-items: center;
    gap: 7px;
    min-height: 36px;
}

.checkbox-control input {
    width: auto;
    min-height: auto;
}

#visible-count {
    margin-left: auto;
    align-self: center;
    color: var(--muted);
}

.table-wrapper {
    max-height: calc(100vh - 230px);
    overflow: auto;
}

table {
    width: 100%;
    min-width: 1500px;
    border-collapse: separate;
    border-spacing: 0;
}

th,
td {
    padding: 10px 12px;
    border-right: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    text-align: left;
    vertical-align: top;
}

thead th {
    position: sticky;
    z-index: 10;
    background: var(--header);
    color: #c9d1d9;
    font-size: 12px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

thead tr:first-child th {
    top: 0;
}

thead tr:nth-child(2) th {
    top: 39px;
}

.player-heading {
    text-align: center;
    color: white;
    font-size: 13px;
}

.player-zero-heading {
    background: #173451;
}

.player-one-heading {
    background: #3a2751;
}

.step-time-heading,
.step-time {
    width: 105px;
    min-width: 105px;
}

.step-time {
    background: inherit;
    color: var(--muted);
    white-space: nowrap;
}

.step-number {
    display: block;
    color: var(--text);
    font-size: 13px;
}

.step-time .day,
.step-time .hour {
    display: block;
    font-size: 11px;
}

tbody tr:nth-child(even) {
    background: rgba(255, 255, 255, 0.018);
}

tbody tr:hover {
    background: rgba(88, 166, 255, 0.08);
}

.farmer-cell {
    min-width: 195px;
}

.hands-cell {
    min-width: 230px;
}

.market-cell {
    min-width: 245px;
}

.balance-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 10px;
    padding: 8px 10px;
    border: 1px solid rgba(227, 179, 65, 0.45);
    border-radius: 7px;
    background: rgba(227, 179, 65, 0.10);
}

.balance-label {
    color: #c9d1d9;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.balance-amount {
    color: var(--gold);
    font-size: 17px;
    font-variant-numeric: tabular-nums;
}

.farmer-position {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    margin-bottom: 10px;
    padding: 0 2px 9px;
    border-bottom: 1px solid var(--border);
    color: var(--muted);
    font-size: 11px;
}

.farmer-position strong {
    color: var(--text);
    font: 12px ui-monospace, Consolas, monospace;
}

.section-label {
    margin-bottom: 5px;
    color: var(--muted);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.hand-card {
    margin-bottom: 8px;
    overflow: hidden;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: rgba(255, 255, 255, 0.025);
}

.hand-card:last-child {
    margin-bottom: 0;
}

.hand-heading {
    padding: 6px 8px;
    border-bottom: 1px solid var(--border);
    background: rgba(255, 255, 255, 0.045);
    color: #c9d1d9;
    font: 12px ui-monospace, Consolas, monospace;
}

.hand-action {
    padding: 7px 8px;
}

.action {
    display: inline-flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 5px;
    padding: 3px 7px;
    border: 1px solid rgba(63, 185, 80, 0.35);
    border-radius: 999px;
    background: rgba(63, 185, 80, 0.10);
    white-space: nowrap;
}

.action strong {
    color: var(--green);
    font-size: 12px;
}

.arguments {
    color: var(--purple);
    font: 12px ui-monospace, SFMono-Regular, Consolas, monospace;
}

.action.pass {
    padding-inline: 0;
    border-color: transparent;
    background: transparent;
}

.action.pass strong,
.empty {
    color: #6e7681;
    font-weight: normal;
}

.action.unknown {
    border-color: #d29922;
}

.market-order {
    margin-bottom: 8px;
    overflow: hidden;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.025);
}

.market-order:last-child {
    margin-bottom: 0;
}

.market-order.buy {
    border-left: 4px solid var(--blue);
}

.market-order.sell {
    border-left: 4px solid var(--orange);
}

.market-order.other {
    border-left: 4px solid var(--muted);
}

.order-heading {
    display: grid;
    gap: 3px;
    padding: 8px 10px;
}

.order-type {
    color: var(--muted);
    font-size: 10px;
    font-weight: 750;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}

.order-description {
    display: flex;
    align-items: baseline;
    gap: 7px;
}

.order-item {
    color: var(--text);
    font-size: 14px;
}

.order-quantity {
    color: var(--purple);
    font: 13px ui-monospace, Consolas, monospace;
}

.order-pricing {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 10px;
    padding: 7px 10px;
    border-top: 1px solid var(--border);
    background: rgba(0, 0, 0, 0.18);
}

.unit-price {
    color: var(--muted);
    font-size: 11px;
}

.total-price {
    color: var(--gold);
    font-size: 14px;
    font-variant-numeric: tabular-nums;
}

.price-unavailable {
    color: var(--muted);
    font-size: 11px;
    font-style: italic;
}

.hidden {
    display: none;
}

.no-results {
    padding: 40px;
    color: var(--muted);
    text-align: center;
}

@media (max-width: 900px) {
    header {
        padding: 20px;
    }

    .market-panel {
        position: static;
        width: auto;
        max-height: none;
        margin-top: 16px;
    }

    #visible-count {
        width: 100%;
        margin-left: 0;
    }

    .table-wrapper {
        max-height: none;
    }
}
</style>
</head>

<body>
<header>
    <h1>__TITLE__ action report</h1>

    <div class="meta">
        <span><strong>__STEP_COUNT__</strong> steps</span>
        <span>Rewards <strong>__REWARDS__</strong></span>
        <span>Status <strong>__STATUSES__</strong></span>
        <span>Source <strong>__SOURCE__</strong></span>
    </div>

    <aside class="market-panel">
        <div class="market-panel-header">
            <strong>Market prices</strong>
            <span id="market-step">Step 0</span>
        </div>

        <div id="market-prices" class="market-prices">
            No market data
        </div>
    </aside>
</header>

<section class="toolbar">
    <label class="control">
        Search
        <input
            id="search"
            type="search"
            placeholder="PLANT, WHEAT, $500..."
        >
    </label>

    <label class="control">
        First step
        <input
            id="first-step"
            type="number"
            min="0"
            max="__LAST_STEP__"
            value="0"
        >
    </label>

    <label class="control">
        Last step
        <input
            id="last-step"
            type="number"
            min="0"
            max="__LAST_STEP__"
            value="__LAST_STEP__"
        >
    </label>

    <label class="checkbox-control">
        <input id="hide-idle" type="checkbox">
        Hide PASS-only steps
    </label>

    <label class="control">
        Jump to step
        <input
            id="jump-step"
            type="number"
            min="0"
            max="__LAST_STEP__"
        >
    </label>

    <button id="reset" type="button">Reset</button>
    <output id="visible-count"></output>
</section>

<div id="table-wrapper" class="table-wrapper">
    <table>
        <thead>
            <tr>
                <th
                    rowspan="2"
                    class="step-time-heading"
                >
                    Step / Time
                </th>

                <th
                    colspan="3"
                    class="player-heading player-zero-heading"
                >
                    Player 0
                </th>

                <th
                    colspan="3"
                    class="player-heading player-one-heading"
                >
                    Player 1
                </th>
            </tr>

            <tr>
                <th>Farmer</th>
                <th>Farmhands</th>
                <th>Market</th>

                <th>Farmer</th>
                <th>Farmhands</th>
                <th>Market</th>
            </tr>
        </thead>

        <tbody id="action-rows">
            __ROWS__
        </tbody>
    </table>

    <div id="no-results" class="no-results hidden">
        No steps match the current filters.
    </div>
</div>

<script>
const rows = [...document.querySelectorAll("#action-rows tr")];
const wrapper = document.querySelector("#table-wrapper");
const search = document.querySelector("#search");
const firstStep = document.querySelector("#first-step");
const lastStep = document.querySelector("#last-step");
const hideIdle = document.querySelector("#hide-idle");
const jumpStep = document.querySelector("#jump-step");
const visibleCount = document.querySelector("#visible-count");
const noResults = document.querySelector("#no-results");
const marketStep = document.querySelector("#market-step");
const marketPrices = document.querySelector("#market-prices");

function updateMarketPanel(row) {
    if (!row) {
        marketStep.textContent = "No step selected";
        marketPrices.textContent = "No market data";
        return;
    }

    marketStep.textContent = `Step ${row.dataset.step}`;

    let prices = {};

    try {
        prices = JSON.parse(row.dataset.market || "{}");
    } catch {
        prices = {};
    }

    const entries = Object.entries(prices).sort(
        ([left], [right]) => left.localeCompare(right)
    );

    if (!entries.length) {
        marketPrices.textContent = "No market data";
        return;
    }

    marketPrices.replaceChildren();

    for (const [item, price] of entries) {
        const element = document.createElement("span");
        element.className = "market-price";

        const name = document.createElement("span");
        name.textContent = item;

        const value = document.createElement("strong");
        value.textContent =
            `$${Number(price).toLocaleString()}`;

        element.append(name, value);
        marketPrices.append(element);
    }
}

function getVisibleRows() {
    return rows.filter(
        row => !row.classList.contains("hidden")
    );
}

function updateMarketFromScroll() {
    const shown = getVisibleRows();

    if (!shown.length) {
        updateMarketPanel(null);
        return;
    }

    const wrapperTop =
        wrapper.getBoundingClientRect().top;

    let closest = shown[0];
    let closestDistance = Infinity;

    for (const row of shown) {
        const distance = Math.abs(
            row.getBoundingClientRect().top
            - wrapperTop
            - 75
        );

        if (distance < closestDistance) {
            closest = row;
            closestDistance = distance;
        }
    }

    updateMarketPanel(closest);
}

function applyFilters() {
    const query =
        search.value.trim().toLowerCase();

    const first =
        firstStep.value === ""
            ? 0
            : Number(firstStep.value);

    const last =
        lastStep.value === ""
            ? Infinity
            : Number(lastStep.value);

    let visible = 0;

    for (const row of rows) {
        const step = Number(row.dataset.step);

        const matches =
            step >= first
            && step <= last
            && (
                !hideIdle.checked
                || row.dataset.idle !== "true"
            )
            && (
                !query
                || row.textContent
                    .toLowerCase()
                    .includes(query)
            );

        row.classList.toggle("hidden", !matches);

        if (matches) {
            visible++;
        }
    }

    visibleCount.textContent =
        `${visible.toLocaleString()} of `
        + `${rows.length.toLocaleString()} steps shown`;

    noResults.classList.toggle(
        "hidden",
        visible !== 0,
    );

    updateMarketFromScroll();
}

let scrollFrame = null;

wrapper.addEventListener("scroll", () => {
    if (scrollFrame !== null) {
        cancelAnimationFrame(scrollFrame);
    }

    scrollFrame = requestAnimationFrame(() => {
        updateMarketFromScroll();
        scrollFrame = null;
    });
});

for (const row of rows) {
    row.addEventListener("mouseenter", () => {
        updateMarketPanel(row);
    });

    row.addEventListener("click", () => {
        updateMarketPanel(row);
    });
}

for (const control of [
    search,
    firstStep,
    lastStep,
    hideIdle,
]) {
    control.addEventListener(
        "input",
        applyFilters,
    );
}

document
    .querySelector("#reset")
    .addEventListener("click", () => {
        search.value = "";
        firstStep.value = 0;
        lastStep.value = __LAST_STEP__;
        hideIdle.checked = false;
        applyFilters();
    });

jumpStep.addEventListener("change", () => {
    const row = document.querySelector(
        `#step-${jumpStep.value}`
    );

    if (row) {
        row.scrollIntoView({
            behavior: "smooth",
            block: "center",
        });

        updateMarketPanel(row);
    }
});

document.addEventListener("keydown", event => {
    if (
        event.key === "/"
        && document.activeElement.tagName !== "INPUT"
    ) {
        event.preventDefault();
        search.focus();
    }
});

applyFilters();
</script>
</body>
</html>
"""

    replacements = {
        "__TITLE__": title,
        "__STEP_COUNT__": str(len(steps)),
        "__REWARDS__": rewards,
        "__STATUSES__": statuses,
        "__SOURCE__": source,
        "__LAST_STEP__": str(last_step),
        "__ROWS__": "".join(rows),
    }

    for placeholder, value in replacements.items():
        page = page.replace(
            placeholder,
            value,
        )

    return page


def parse_args() -> argparse.Namespace:
    script_directory = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Create a detailed HTML report of both players' "
            "actions, positions, money, and market prices."
        )
    )

    parser.add_argument(
        "replay",
        nargs="?",
        type=Path,
        default=script_directory / "replay.json",
        help=(
            "Replay JSON; defaults to replay.json "
            "beside this script."
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Output HTML; defaults to actions.html "
            "beside the replay."
        ),
    )

    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Create the report without opening it.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    replay_path = args.replay.expanduser().resolve()

    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else replay_path.with_name("actions.html")
    )

    with replay_path.open(
        "r",
        encoding="utf-8",
    ) as replay_file:
        replay = json.load(replay_file)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        build_actions_html(
            replay,
            replay_path.name,
        ),
        encoding="utf-8",
    )

    print(f"Wrote action report to {output_path}")

    if not args.no_open:
        webbrowser.open(output_path.as_uri())


if __name__ == "__main__":
    main()