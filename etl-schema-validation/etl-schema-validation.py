def validate_records(records, schema):
    type_map = {'int': int, 'str': str, 'float': float}
    results = []

    for idx, record in enumerate(records):
        errors = []

        for field in schema:
            col      = field['column']
            expected = field['type']
            nullable = field.get('nullable', True)

            # Missing field
            if col not in record:
                errors.append(f"{col}: missing")
                continue

            value = record[col]

            # Null check
            if value is None:
                if not nullable:
                    errors.append(f"{col}: null")
                continue

            # Type check
            actual_type = type(value)
            if expected == 'float':
                if actual_type not in (int, float):
                    errors.append(f"{col}: expected float, got {actual_type.__name__}")
                    continue
            else:
                if actual_type is not type_map[expected]:
                    errors.append(f"{col}: expected {expected}, got {actual_type.__name__}")
                    continue

            # Range check
            if ('min' in field and value < field['min']) or \
               ('max' in field and value > field['max']):
                errors.append(f"{col}: out of range")

        results.append((idx, len(errors) == 0, errors))

    return results