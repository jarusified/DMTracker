import { 
    FETCH_REUSE, 
    FETCH_COMM, 
    FETCH_EXPERIMENTS, 
    FETCH_CCT, 
    FETCH_TIMELINE, 
    FETCH_METRICS, 
    FETCH_KERNELS,
    FETCH_ENSEMBLE,
    TEST_FETCH_JSON,
    UPDATE_EXPERIMENT,
    UPDATE_KERNEL,
    UPDATE_METRIC,
} from './helpers/types';

const initialState = {
    reuse: {},
    experiments: [],
    selected_experiment: '',
    cct: {},
    timeline: {},
    runtime_metrics: {},
    transfer_metrics: {},
    atts: {},
    kernel_metrics: {},
    kernels: [],
    selected_kernel: '',
    metrics: [],
    selected_metric: '',
    testJSON: null,
    comm_matrix: {},
};

export default function Reducer(state=initialState, action){
    switch (action.type) {
        case FETCH_EXPERIMENTS:
            return {
                ...state,
                experiments: action.payload.experiments,
                selected_experiment: action.payload.experiments[0],
            }
        case FETCH_REUSE:
            return {
                ...state,
                reuse: action.payload.data,
            }
        case FETCH_CCT:
            return {
                ...state,
                cct: action.payload,
            }
        case FETCH_TIMELINE:
            return {
                ...state,
                timeline: action.payload,
            }
        case FETCH_ENSEMBLE:
            return {
                ...state,
                kernel_metrics: action.payload.kernel_metrics,
                runtime_metrics: action.payload.runtime_metrics,
                transfer_metrics: action.payload.transfer_metrics,
                atts: action.payload.atts,
                
            }
        case FETCH_METRICS:
            return {
                ...state,
                metrics: action.payload.metrics,
                selected_metric: action.payload.metrics[0],
            }
        case FETCH_KERNELS:
            return {
                ...state,
                kernels: action.payload.kernels,
                selected_kernel: action.payload.kernels[0],
            }
        case FETCH_COMM:
            return {
                ...state,
                comm_matrix: action.payload,
            }
        case TEST_FETCH_JSON:
            return {
                ...state,
                testJSON: action.payload,
            }
        case UPDATE_EXPERIMENT:
            return {
                ...state,
                selected_experiment: action.payload,
            }
        case UPDATE_KERNEL:
            return {
                ...state,
                selected_kernel: action.payload,
            }
        case UPDATE_METRIC: 
            return {
                ...state,
                selected_metric: action.payload,
            }
        default:
            return state;
    }
}