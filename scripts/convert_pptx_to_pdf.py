"""Convert PPTX to PDF using PowerPoint COM automation (Windows only)."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import win32com.client  # type: ignore
except ImportError:
    print("pywin32 not installed", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    pptx = Path(sys.argv[1])
    pdf = Path(sys.argv[2])
    pdf.parent.mkdir(parents=True, exist_ok=True)

    ppt = win32com.client.DispatchEx("PowerPoint.Application")
    ppt.Visible = True
    pres = ppt.Presentations.Open(str(pptx.absolute()), False, False, False)
    pres.SaveAs(str(pdf.absolute()), 32)  # 32 = ppSaveAsPDF
    pres.Close()
    ppt.Quit()
    print(f"Wrote {pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
