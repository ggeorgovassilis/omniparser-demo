#!/usr/bin/env bash
# browser-guard.sh — PreToolUse hook: enforces tool access policy for visual_tester agent
#
# Layers (evaluated in order):
#   1. Block ANY tool referencing .github/hooks/ or .github/agents/
#   2. Guard run_in_terminal: allow only browser_cli.py + diagnostic utils
#   3. Guard send_to_terminal: empty (Enter) only, blocks everything else
#   4. Block subagent/runSubagent invocations
#   5. Pass through all other tools
#
# Reads tool invocation JSON from stdin → returns permission decision on stdout.

set -euo pipefail
INPUT=$(cat)

TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // .toolName // ""')
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // .tool_input.cmd // ""')

# ── LAYER 1: Block ANY tool targeting protected paths ────────────────────────
# Recursively scan ALL string values in tool_input for .github/hooks or .github/agents.
# Uses [.] instead of \. to avoid jq/shell escape nightmares.
PROTECTED_REGEX='[.]github/hooks|[.]github/agents'

if echo "$INPUT" | jq -e '.tool_input // {} | .. | strings | select(test("'"$PROTECTED_REGEX"'"))' > /dev/null 2>&1; then
    MATCHED=$(echo "$INPUT" | jq -r '[.tool_input // {} | .. | strings | select(test("'"$PROTECTED_REGEX"'"))] | join("; ")' 2>/dev/null || echo "protected path")
    echo "{\"permissionDecision\":\"deny\",\"permissionDecisionReason\":\"Modifying protected paths (.github/hooks/ or .github/agents/) is forbidden. Matched: $MATCHED\"}"
    exit 0
fi

# ── LAYER 4: Block subagent invocations ─────────────────────────────────────
if [ "$TOOL_NAME" = "runSubagent" ] || [ "$TOOL_NAME" = "agent" ]; then
    echo "{\"permissionDecision\":\"deny\",\"permissionDecisionReason\":\"Subagent delegation is disabled for visual_tester. No workarounds.\"}"
    exit 0
fi

# ── LAYER 2: Guard run_in_terminal ──────────────────────────────────────────
if [ "$TOOL_NAME" = "run_in_terminal" ] || [ "$TOOL_NAME" = "execute/runInTerminal" ]; then
    PREFIX="docker compose exec browser-env python /app/poc/browser_cli.py"
    PREFIX_ALT="docker compose exec -T browser-env python /app/poc/browser_cli.py"
    ALLOWED_CMDS="goto observe click type tabs tab"

    ALLOWED_UTILS=(
        "curl -s http://localhost:8000/health"
        "curl -s http://omniparser:8000/health"
        "docker compose ps"
        "docker compose logs"
        "echo "
        "find "
        "wc "
        "ls "
    )

    CORE_CMD=$(echo "$COMMAND" | head -1 | sed 's/^[[:space:]]*//')

    for util in "${ALLOWED_UTILS[@]}"; do
        if [[ "$CORE_CMD" == "$util"* ]]; then
            echo '{"permissionDecision":"allow"}'
            exit 0
        fi
    done

    if [[ "$CORE_CMD" == "$PREFIX "* ]] || [[ "$CORE_CMD" == "$PREFIX_ALT "* ]]; then
        if [[ "$CORE_CMD" == "$PREFIX "* ]]; then
            AFTER="${CORE_CMD#$PREFIX }"
        else
            AFTER="${CORE_CMD#$PREFIX_ALT }"
        fi
        SUBCMD=$(echo "$AFTER" | awk '{print $1}')
        for allowed in $ALLOWED_CMDS; do
            if [ "$SUBCMD" = "$allowed" ]; then
                echo '{"permissionDecision":"allow"}'
                exit 0
            fi
        done
        echo "{\"permissionDecision\":\"deny\",\"permissionDecisionReason\":\"Browser CLI subcommand '$SUBCMD' not allowed. Permitted: $ALLOWED_CMDS\"}"
        exit 0
    fi

    echo "{\"permissionDecision\":\"deny\",\"permissionDecisionReason\":\"Only browser_cli.py commands are permitted. Got: $CORE_CMD\"}"
    exit 0
fi

# ── LAYER 3: Guard send_to_terminal ─────────────────────────────────────────
if [ "$TOOL_NAME" = "send_to_terminal" ] || [ "$TOOL_NAME" = "execute/sendToTerminal" ]; then
    SEND_CMD=$(echo "$INPUT" | jq -r '.tool_input.command // ""')
    if [ -z "$(echo "$SEND_CMD" | tr -d '[:space:]')" ]; then
        echo '{"permissionDecision":"allow"}'
        exit 0
    fi
    echo "{\"permissionDecision\":\"deny\",\"permissionDecisionReason\":\"send_to_terminal with non-empty command blocked. Use browser_cli.py instead.\"}"
    exit 0
fi

# ── LAYER 5: Pass through everything else ───────────────────────────────────
echo '{"permissionDecision":"allow"}'
exit 0
