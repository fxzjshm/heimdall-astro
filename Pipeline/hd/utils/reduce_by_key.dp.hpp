#pragma once

#include <boost/compute/container/vector.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/memory/local_buffer.hpp>

namespace boost::compute::detail {

/// \internal_
///
/// Perform final reduction by key. Each work item:
/// 1. Perform local work-group reduction (Hillis/Steele scan)
/// 2. Add carry-in (if keys are right)
/// 3. Save reduced value if next key is different than processed one
template<class InputKeyIterator, class InputValueIterator,
         class OutputKeyIterator, class OutputValueIterator,
         class OutputValueIterator2, class BinaryFunction>
inline void final_reduction(InputKeyIterator keys_first,
                            InputValueIterator values_first,
                            OutputKeyIterator keys_result,
                            OutputValueIterator values_result,
                            size_t count,
                            BinaryFunction function,
                            vector<uint_>::iterator new_keys_first,
                            vector<uint_>::iterator carry_in_keys_first,
                            OutputValueIterator2 carry_in_values_first,
                            size_t carry_in_size,
                            size_t work_group_size,
                            command_queue &queue)
{
    typedef typename
        std::iterator_traits<OutputValueIterator>::value_type value_out_type;

    detail::meta_kernel k("reduce_by_key_with_scan_final_reduction");
    k.add_set_arg<const uint_>("count", uint_(count));
    size_t local_keys_arg = k.add_arg<uint_ *>(memory_object::local_memory, "lkeys");
    size_t local_vals_arg = k.add_arg<value_out_type *>(memory_object::local_memory, "lvals");

    k <<
        k.decl<const uint_>("gid") << " = get_global_id(0);\n" <<
        k.decl<const uint_>("wg_size") << " = get_local_size(0);\n" <<
        k.decl<const uint_>("lid") << " = get_local_id(0);\n" <<
        k.decl<const uint_>("group_id") << " = get_group_id(0);\n" <<

        k.decl<uint_>("key") << ";\n" <<
        k.decl<value_out_type>("value") << ";\n"

        "if(gid < count){\n" <<
            k.var<uint_>("key") << " = " <<
                new_keys_first[k.var<const uint_>("gid")] << ";\n" <<
            k.var<value_out_type>("value") << " = " <<
                values_first[k.var<const uint_>("gid")] << ";\n" <<
            "lkeys[lid] = key;\n" <<
            "lvals[lid] = value;\n" <<
        "}\n" <<

        // Hillis/Steele scan
        k.decl<value_out_type>("result") << " = value;\n" <<
        k.decl<uint_>("other_key") << ";\n" <<
        k.decl<value_out_type>("other_value") << ";\n" <<

        "for(" << k.decl<uint_>("offset") << " = 1; " <<
                 "offset < wg_size ; offset *= 2){\n"
        "    barrier(CLK_LOCAL_MEM_FENCE);\n" <<
        "    if(lid >= offset) {\n" <<
        "        other_key = lkeys[lid - offset];\n" <<
        "        if(other_key == key){\n" <<
        "            other_value = lvals[lid - offset];\n" <<
        "            result = " << function(k.var<value_out_type>("result"),
                                            k.var<value_out_type>("other_value")) << ";\n" <<
        "        }\n" <<
        "    }\n" <<
        "    barrier(CLK_LOCAL_MEM_FENCE);\n" <<
        "    lvals[lid] = result;\n" <<
        "}\n" <<

        "if(gid >= count) {\n return;\n};\n" <<

        k.decl<const bool>("save") << " = (gid < (count - 1)) ?"
                                   << new_keys_first[k.var<const uint_>("gid + 1")] << " != key" <<
                                   ": true;\n" <<

        // Add carry in
        k.decl<uint_>("carry_in_key") << ";\n" <<
        "if(group_id > 0 && save) {\n" <<
        "    carry_in_key = " << carry_in_keys_first[k.var<const uint_>("group_id - 1")] << ";\n" <<
        "    if(key == carry_in_key){\n" <<
        "        other_value = " << carry_in_values_first[k.var<const uint_>("group_id - 1")] << ";\n" <<
        "        result = " << function(k.var<value_out_type>("result"),
                                        k.var<value_out_type>("other_value")) << ";\n" <<
        "    }\n" <<
        "}\n" <<

        // Save result only if the next key is different or it's the last element.
        "if(save){\n" <<
        keys_result[k.var<uint_>("key")] << " = " << keys_first[k.var<const uint_>("gid")] << ";\n" <<
        values_result[k.var<uint_>("key")] << " = result;\n" <<
        "}\n"
        ;

    size_t work_groups_no = static_cast<size_t>(
        std::ceil(float(count) / work_group_size)
    );

    const context &context = queue.get_context();
    kernel kernel = k.compile(context);
    kernel.set_arg(local_keys_arg, local_buffer<uint_>(work_group_size));
    kernel.set_arg(local_vals_arg, local_buffer<value_out_type>(work_group_size));

    queue.enqueue_1d_range_kernel(kernel,
                                  0,
                                  work_groups_no * work_group_size,
                                  work_group_size);
}

}

#include <boost/compute/algorithm/reduce_by_key.hpp>