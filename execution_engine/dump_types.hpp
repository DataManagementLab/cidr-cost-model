#pragma once

enum op_type
{
    INVALID_OP = -1,
    SELECT_BRANCH = 0,
    SELECT_NO_BRANCH,
    PROBE,
    BUILD
};

enum pred_type
{
    INVALID_PRED = -1,
    GREATER = 0,
    LESS,
    EQUAL,
    TRUE,
    HALF
};

enum pipeline_type
{
    INVALID_PIPELINE = -1,
    SEL_BUILD = 0,
    SEL_PROBE,
    SEL_PROBE_BUILD
};