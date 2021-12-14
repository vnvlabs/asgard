#ifndef _VnV_adaptivity
#define _VnV_adaptivity

#include "../adapt.hpp"
#include "VnV.h"

INJECTION_REGISTRATION(MeshWatcher);

/**
 * Asgard Adaptivity Tracking.
 * ===========================
 *
 * The final grid had :vnv:`Elements[-1]` elements.
 *
 * .. vnv-chart::
 *
 *    {
 *       "type" : "line",
 *       "data" : {
 *          "labels" : {{as_json(Labels)}},
 *          "datasets" : [{
 *             "label": "Number of elements",
 *             "backgroundColor": "rgb(255, 99, 132)",
 *             "borderColor": "rgb(255, 99, 132)",
 *             "data": {{as_json(Elements)}}
 *           }]
 *       },
 *       "options" : {
 *           "animation" : false,
 *           "responsive" : true,
 *           "title" : { "display" : true,
 *                       "text" : "The Number of elements in the adaptive grid."
 *                     },
 *          "scales": {
 *             "yAxes": [{
 *               "scaleLabel": {
 *                 "display": true,
 *                 "labelString": "Number of Elements"
 *               }
 *            }],
 *            "xAxes": [{
 *              "scaleLabel": {
 *                 "display":true,
 *                 "labelString": "Injection Point Stage"
 *               }
 *            }]
 *          }
 *       }
 *    }
 *
 *
 */

INJECTION_TEST(MeshWatcher, AdaptivityTracking) {

  auto &grid = GetRef("adaptive_grid",adapt::distributed_grid<double>);
  engine->Put("Elements", grid.size());
  engine->Put("Labels", stageId);

  return SUCCESS;
}

#endif
