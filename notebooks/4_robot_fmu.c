#include "fmi3Functions.h"
#include "fmi3FunctionTypes.h"
#include <math.h>

// ------------------------------ fully functional ------------------------------------

typedef struct Robot
{
    // parameters
    double V_abs; // supply voltage
    double K;     //  torque coefficient
    double g;     //  gravitational acceleration
    double b;     //  motor shaft friction
    double m;     //  mass of joint
    double R;     //  electrical resistance of coil
    double L;     //  motor inductance
    double l;     //  length of joint
    double J;     //  moment of intertia
    // states
    double θ; // angle
    double ω; // velocity
    // inputs
    double u; // controller output

} Robot;

fmi3Instance fmi3InstantiateCoSimulation(
    fmi3String instanceName,
    fmi3String instantiationToken,
    fmi3String resourcePath,
    fmi3Boolean visible,
    fmi3Boolean loggingOn,
    fmi3Boolean eventModeUsed,
    fmi3Boolean earlyReturnAllowed,
    const fmi3ValueReference requiredIntermediateVariables[],
    size_t nRequiredIntermediateVariables,
    fmi3InstanceEnvironment instanceEnvironment,
    fmi3LogMessageCallback logMessage,
    fmi3IntermediateUpdateCallback intermediateUpdate)
{
    return malloc(sizeof(Robot));
}

fmi3Float64 *vref_to_float64(Robot *m, fmi3ValueReference valueReference)
{
    switch (valueReference)
    {
    case 0:
        return &(m->V_abs);
        break;
    case 1:
        return &(m->K);
        break;
    case 2:
        return &(m->g);
        break;
    case 3:
        return &(m->b);
        break;
    case 4:
        return &(m->m);
        break;
    case 5:
        return &(m->R);
        break;
    case 6:
        return &(m->L);
        break;
    case 7:
        return &(m->l);
        break;
    case 8:
        return &(m->J);
        break;
    case 9:
        return &(m->θ);
        break;
    case 10:
        return &(m->ω);
        break;
    case 11:
        return &(m->u);
        break;
    default:
        return NULL;
    }
}

fmi3Status fmi3GetFloat64(fmi3Instance instance,
                          const fmi3ValueReference valueReferences[],
                          size_t nValueReferences,
                          fmi3Float64 values[],
                          size_t nValues)
{
    Robot *m = (Robot *)instance;

    for (int i = 0; i < nValueReferences; ++i)
    {
        fmi3ValueReference valueReference = valueReferences[i];

        values[valueReference] = *vref_to_float64(m, valueReference);
    }
    return fmi3OK;
}

fmi3Status fmi3GetFloat64(fmi3Instance instance,
                          const fmi3ValueReference valueReferences[],
                          size_t nValueReferences,
                          fmi3Float64 values[],
                          size_t nValues)
{
    Robot *m = (Robot *)instance;

    for (int i = 0; i < nValueReferences; ++i)
    {
        fmi3ValueReference valueReference = valueReferences[i];

        *vref_to_float64(m, valueReference) = values[valueReference];
    }
    return fmi3OK;
}

fmi3Status fmi3DoStep(fmi3Instance instance,
                      fmi3Float64 currentCommunicationPoint,
                      fmi3Float64 communicationStepSize,
                      fmi3Boolean noSetFMUStatePriorToCurrentPoint,
                      fmi3Boolean *eventHandlingNeeded,
                      fmi3Boolean *terminateSimulation,
                      fmi3Boolean *earlyReturn,
                      fmi3Float64 *lastSuccessfulTime)
{
    Robot *m = (Robot *)instance;

    double time_internal = 0.0;
    double internal_step_size = 1 / 1024;

    while (time_internal != communicationStepSize)
    {
        double dθdt = m->ω;
        double dωdt = -m->g / m->l * sin(m->θ);
        m->θ += dθdt;
        m->ω += dωdt;
        time_internal += internal_step_size;
    }

    return fmi3OK;
}

// ------------------------------ reduced ------------------------------------

fmi3Instance _fmi3InstantiateCoSimulation()
{
    return malloc(sizeof(Robot));
}

fmi3Status _fmi3DoStep(void *instance,
                       double currentCommunicationPoint,
                       double communicationStepSize,
                       // parameters omitted for brevity
                       double *lastSuccessfulTime)
{
    Robot *m = (Robot *)instance;

    double time_internal = 0.0;
    double internal_step_size = 1 / 1024;

    while (time_internal != communicationStepSize)
    {
        double dθdt = m->ω;
        double dωdt = -m->g / m->l * sin(m->θ);
        m->θ += dθdt;
        m->ω += dωdt;
        time_internal += internal_step_size;
    }

    return fmi3OK;
}
