"""Small immutable-stage decisions, independently testable without data/model access."""


def solve_action(started, coefficients, receipt):
    if coefficients and receipt:
        return 'verify_existing'
    if coefficients:
        return 'recover_receipt_without_solve'
    if started or receipt:
        raise RuntimeError('A prior solve may have started without preserved coefficients; do not repeat automatically')
    return 'solve_once'


def evaluation_action(completed, attempts):
    if completed:
        raise RuntimeError('Validation is complete; repeated scoring is forbidden')
    if attempts:
        raise RuntimeError('An earlier validation attempt exists; review interruption before any retry')
    return 'evaluate_once'
