import uuid


def generate_8_digit_uuid():
    full_uuid = uuid.uuid4()
    hex_str = full_uuid.hex[:8]
    return hex_str


def generate_4_digit_uuid():
    full_uuid = uuid.uuid4()
    hex_str = full_uuid.hex[:4]
    return hex_str
