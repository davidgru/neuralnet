#pragma once

#include "dataset.h"


uint32_t dataset_create_train_and_test(
    const dataset_impl_t* impl,
    const dataset_create_info_t* train_config,
    const dataset_create_info_t* test_config,
    bool normalize,
    dataset_t* out_train_set,
    dataset_t* out_test_set
);
