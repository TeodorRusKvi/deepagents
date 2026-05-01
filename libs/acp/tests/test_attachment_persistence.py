from __future__ import annotations

import base64
from pathlib import Path

from acp.schema import BlobResourceContents, EmbeddedResourceContentBlock, ImageContentBlock

from deepagents_acp.server import AgentServerACP


def _make_server(tmp_path: Path) -> AgentServerACP:
    server = AgentServerACP.__new__(AgentServerACP)
    server._cwd = tmp_path.as_posix()
    return server


def test_persist_image_block_writes_attachment(tmp_path) -> None:
    server = _make_server(tmp_path)
    block = ImageContentBlock(
        type="image",
        mime_type="image/png",
        data=base64.b64encode(b"image-bytes").decode("ascii"),
    )

    server._persist_image_block(block, session_id="session-1")

    attachments = list((tmp_path / "attachments" / "session-1").glob("*.png"))
    assert len(attachments) == 1
    assert attachments[0].read_bytes() == b"image-bytes"


def test_persist_embedded_resource_block_writes_pdf(tmp_path) -> None:
    server = _make_server(tmp_path)
    block = EmbeddedResourceContentBlock(
        type="resource",
        resource=BlobResourceContents(
            uri="file:///tmp/report.pdf",
            mime_type="application/pdf",
            blob=base64.b64encode(b"pdf-bytes").decode("ascii"),
        ),
    )

    server._persist_embedded_resource_block(block, session_id="session-2")

    attachments = list((tmp_path / "attachments" / "session-2").glob("*.pdf"))
    assert len(attachments) == 1
    assert attachments[0].read_bytes() == b"pdf-bytes"
