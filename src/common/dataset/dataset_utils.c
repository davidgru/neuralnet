#include "dataset_utils.h"


uint32_t dataset_create_train_and_test(
    const dataset_impl_t* impl,
    const dataset_create_info_t* config,
    bool normalize,
    dataset_t* out_train_set,
    dataset_t* out_test_set
)
{
    uint32_t status = dataset_create(out_train_set, impl, config, TRAIN_SET, normalize, NULL);
    if (status != 0) {
        return status;
    }

    const dataset_statistics_t* train_statistics = dataset_get_statistics(*out_train_set);
    status = dataset_create(out_test_set, impl, config, TEST_SET, normalize, train_statistics);
    if (status != 0) {
        dataset_destroy(*out_train_set);
        return 1;
    }
    return 0;
}
