"""
Copyright 2018-2019 CS Systèmes d'Information

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""
import datetime


def sk_mdl_to_tdt(mdl, feature_names, ls_name="No name"):
    """
    Convert a scikit learn DecisionTreeClassifier model into Temporal Decision Tree format

    :param feature_names: Name of the various features used to fill the output format
    :type feature_names: list

    :param ls_name: Learning set name used to fill the output format
    :type ls_name: str

    :param mdl: scikit learn model
    :type mdl: DecisionTreeClassifier

    :return: TDT object JSON compatible
    :rtype: dict
    """

    # Initialize the output
    tdt = {
        "header": {
            "TDTFormatVersion": "2.48",
            "date": str(datetime.datetime.now().strftime("%Y-%m-%d")),
            "builder": "IKATS",
            "learningSet": ls_name,
            "source": "IKATS"
        },
        "tree": []
    }

    # Building the node list
    for node_id in range(mdl.tree_.node_count):

        # Default values for node
        node = {
            "node": node_id,
            "name": feature_names[mdl.tree_.feature[node_id]],
            "type": "variable",
            "evaluation": [mdl.criterion, mdl.tree_.impurity[node_id]],
            "statistics": [x for x in zip(list(mdl.classes_), list(mdl.tree_.value[node_id][0])) if x[1] != 0],
            "description": {
                "variable": feature_names[mdl.tree_.feature[node_id]],
                "type": "float",
                "criteria": []
            }
        }

        # children found
        if mdl.tree_.children_left[node_id] != -1 or mdl.tree_.children_right[node_id] != -1:
            node["description"]["criteria"].extend([{
                "test": "<=",
                "value": mdl.tree_.threshold[node_id],
                "child": int(mdl.tree_.children_left[node_id])
            }, {
                "test": ">",
                "value": mdl.tree_.threshold[node_id],
                "child": int(mdl.tree_.children_right[node_id])
            }])

        # No children found
        if len(node["description"]["criteria"]) == 0:
            # No description to set
            del (node["description"])
            # The node name shall be set to the class name
            node["name"] = node["statistics"][0][0]
            node["type"] = "leaf"

        tdt['tree'].append(node)

    return tdt
