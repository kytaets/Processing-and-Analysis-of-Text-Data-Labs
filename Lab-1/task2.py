
import re
from pathlib import Path

INPUT_PATH = Path("text11.txt")
PHONES_OUT_PATH = Path("phones.txt")
MASK_CHAR = "X"

PHONE_RE = re.compile(r"""
(?<!\w)
\(?\+?\d
[\d\-\s\(\)]{7,}
\d
(?!\w)
""", re.VERBOSE)


def mask_phone_keep_first_two_digits(phone: str, mask_char: str = "X") -> str:
    result = []
    digit_count = 0

    for ch in phone:
        if ch.isdigit():
            digit_count += 1
            if digit_count <= 2:
                result.append(ch)
            else:
                result.append(mask_char)
        else:
            result.append(ch)

    return "".join(result)


def main():
    text = INPUT_PATH.read_text(encoding="utf-8")

    phones = PHONE_RE.findall(text)

    PHONES_OUT_PATH.write_text("\n".join(phones), encoding="utf-8")

    def replace_function(match):
        return mask_phone_keep_first_two_digits(match.group(0), MASK_CHAR)

    masked_text = PHONE_RE.sub(replace_function, text)

    print("===== Changed Text =====")
    print(masked_text)

    print("\nNumbers found:", len(phones))
    print("Numbers are saved in phones.txt")


if __name__ == "__main__":
    main()
