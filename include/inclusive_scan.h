#pragma once

#if !defined(RXCPP_OPERATORS_RX_SCAN_HPP)
#define RXCPP_OPERATORS_RX_SCAN_HPP

#include <rxcpp/rx-includes.hpp>

struct inclusive_scan_tag {
    template<class Included>
    struct include_header{
        static_assert(Included::value, "missing include: please #include <rxcpp/operators/rx-scan.hpp>");
    };
};

namespace rxcpp {

namespace operators {

namespace detail {

template<class... AN>
struct inclusive_scan_invalid_arguments {};

template<class... AN>
struct inclusive_scan_invalid : public rxo::operator_base<inclusive_scan_invalid_arguments<AN...>> {
    using type = observable<inclusive_scan_invalid_arguments<AN...>, inclusive_scan_invalid<AN...>>;
};
template<class... AN>
using inclusive_scan_invalid_t = typename inclusive_scan_invalid<AN...>::type;

template<class T, class Observable, class Accumulator, class Seed>
struct inclusive_scan : public operator_base<rxu::decay_t<Seed>>
{
    using source_type = rxu::decay_t<Observable>;
    using accumulator_type = rxu::decay_t<Accumulator>;
    using seed_type = rxu::decay_t<Seed>;

    struct inclusive_scan_initial_type
    {
        inclusive_scan_initial_type(source_type o, accumulator_type a, seed_type s)
            : source(std::move(o))
            , accumulator(std::move(a))
            , seed(s)
        {
        }
        source_type source;
        accumulator_type accumulator;
        seed_type seed;
    };
    inclusive_scan_initial_type initial;

    inclusive_scan(source_type o, accumulator_type a, seed_type s)
        : initial(std::move(o), a, s)
    {
    }

    template<class Subscriber>
    void on_subscribe(Subscriber o) const {
        struct inclusive_scan_state_type
            : public inclusive_scan_initial_type
            , public std::enable_shared_from_this<inclusive_scan_state_type>
        {
            inclusive_scan_state_type(inclusive_scan_initial_type i, Subscriber scrbr)
                : inclusive_scan_initial_type(i)
                , result(inclusive_scan_initial_type::seed)
                , out(std::move(scrbr))
            {
            }
            seed_type result;
            Subscriber out;
        };
        auto state = std::make_shared<inclusive_scan_state_type>(initial, std::move(o));
        state->source.subscribe(
            state->out,
        // on_next
            [state](auto&& t) {
                state->result = state->accumulator(std::move(state->result), std::forward<decltype(t)>(t));
                state->out.on_next(state->result);
            },
        // on_error
            [state](rxu::error_ptr e) {
                state->out.on_error(e);
            },
        // on_completed
            [state]() {
                state->out.on_completed();
            }
        );
    }
};

}

/*! @copydoc rx-inclusive_scan.hpp
*/
template<class... AN>
auto inclusive_scan(AN&&... an)
    ->     operator_factory<inclusive_scan_tag, AN...> {
    return operator_factory<inclusive_scan_tag, AN...>(std::make_tuple(std::forward<AN>(an)...));
}

}

template<>
struct member_overload<inclusive_scan_tag>
{
    template<class Observable, class Seed, class Accumulator,
        class Enabled = rxu::enable_if_all_true_type_t<
            is_observable<Observable>,
            is_accumulate_function_for<rxu::value_type_t<Observable>, rxu::decay_t<Seed>, rxu::decay_t<Accumulator>>>,
        class SourceValue = rxu::value_type_t<Observable>,
        class InclusiveScan = rxo::detail::inclusive_scan<SourceValue, rxu::decay_t<Observable>, rxu::decay_t<Accumulator>, rxu::decay_t<Seed>>,
        class Value = rxu::value_type_t<InclusiveScan>,
        class Result = observable<Value, InclusiveScan>>
    static Result member(Observable&& o, Seed s, Accumulator&& a) {
        return Result(InclusiveScan(std::forward<Observable>(o), std::forward<Accumulator>(a), s));
    }

    template<class... AN>
    static operators::detail::inclusive_scan_invalid_t<AN...> member(AN...) {
        std::terminate();
        return {};
        static_assert(sizeof...(AN) == 10000, "inclusive_scan takes (Seed, Accumulator); Accumulator must be a function with the signature Seed(Seed, T)");
    }
};

}

#endif
