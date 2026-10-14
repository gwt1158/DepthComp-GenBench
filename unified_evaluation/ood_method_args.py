"""Fail-closed forwarding of method-only flags through the OOD-16 launcher.

``eval_ood16`` intentionally knows nothing about model-specific arguments.  A
repository wrapper must consume its source-domain flag and forward it to every
per-domain evaluation subprocess via ``--eval-extra-arg``.  Passing the method
flag straight to the generic parser is an error and, worse, omitting it lets an
OOD target accidentally influence source-model preprocessing.
"""


def forward_method_args(argv, *, required, optional=(), optional_boolean=()):
    """Return ``(selected, forwarded)`` for method-specific value flags.

    ``required`` maps a flag name to its allowed string values. ``optional`` is
    an iterable of additional value-taking flags. Unknown arguments are kept
    unchanged for the generic OOD-16 parser. Duplicate, empty, missing, or
    invalid method flags fail before any dataset subprocess is launched.
    """
    required = {str(name): tuple(str(value) for value in values)
                for name, values in required.items()}
    optional = tuple(str(name) for name in optional)
    optional_boolean = tuple(str(name) for name in optional_boolean)
    method_flags = tuple(required) + optional + optional_boolean
    selected = {}
    forwarded = []
    index = 0

    while index < len(argv):
        token = str(argv[index])
        matched = next((name for name in method_flags
                        if token == name or token.startswith(name + "=")), None)
        if matched is None:
            forwarded.append(token)
            index += 1
            continue
        if matched in selected:
            raise SystemExit(f"{matched} may be specified only once")
        if matched in optional_boolean:
            if token != matched:
                raise SystemExit(f"{matched} is a boolean flag and takes no value")
            selected[matched] = True
            forwarded.append(f"--eval-extra-arg={matched}")
            index += 1
            continue
        if token == matched:
            if index + 1 >= len(argv):
                raise SystemExit(f"{matched} requires a value")
            value = str(argv[index + 1])
            index += 2
        else:
            value = token.split("=", 1)[1]
            index += 1
        if not value or value.startswith("--"):
            raise SystemExit(f"{matched} requires a non-empty value")
        allowed = required.get(matched)
        if allowed is not None and value not in allowed:
            raise SystemExit(
                f"{matched} must be one of {', '.join(allowed)}; got {value!r}")
        selected[matched] = value
        forwarded.append(f"--eval-extra-arg={matched}={value}")

    missing = [name for name in required if name not in selected]
    if missing:
        raise SystemExit(
            "OOD evaluation requires explicit source-model selection: "
            + ", ".join(missing))
    return selected, forwarded
