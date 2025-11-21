import slicer
import os
import vtk


referenceVolumeName = "A1_grayT2"
outputDir = "/Users/vivienyu/Desktop/opt_proj/test3/nii_exports"

folderNamesOfInterest = [
    "left temporal lobe",
    "subcortex of left cerebral hemisphere",
    "left lateral ventricle",
    "left frontal lobe",
    "left insula",
    "left parietal lobe",
    "left limbic lobe",
    "left occipital lobe",
    "right temporal lobe",
    "right frontal lobe",
    "right occipital lobe",
    "right parietal lobe",
    "right limbic lobe",
    "right insula",
    "subcortex of right cerebral hemisphere",
]

extraModelNames = [
    "Model_390_right_subthalamic_nucleus",
    "Model_391_left_subthalamic_nucleus",
    "Model_7_white_matter_of_left_hemisphere_of_cerebellum",
    "Model_46_white_matter_of_right_hemisphere_of_cerebellum",
    "Model_3004_corpus_callosum",
    "Model_4_left_lateral_ventricle",
    "Model_5_temporal_horn_of_left_lateral_ventricle",
    "Model_43_right_lateral_ventricle",
    "Model_44_temporal_horn_of_right_lateral_ventricle",
    "Model_19_aqueduct",
    "Model_15_fourth_ventricle",
    "Model_24_third_ventricle",
]

if not os.path.isdir(outputDir):
    os.makedirs(outputDir)

referenceVolumeNode = slicer.util.getNode(referenceVolumeName)

shNode = slicer.mrmlScene.GetSubjectHierarchyNode()

def getModelNodesInFolder(folderName):
    """Return list of vtkMRMLModelNode under a Subject Hierarchy folder."""
    models = []
    folderItemID = shNode.GetItemByName(folderName)
    if not folderItemID:
        print(f"[WARN] Folder '{folderName}' not found in Subject Hierarchy, skipping.")
        return models

    childItemIDs = vtk.vtkIdList()
    shNode.GetItemChildren(folderItemID, childItemIDs, True)  # recursive=True

    for i in range(childItemIDs.GetNumberOfIds()):
        childItemID = childItemIDs.GetId(i)
        dataNode = shNode.GetItemDataNode(childItemID)
        if dataNode and dataNode.IsA("vtkMRMLModelNode"):
            name = dataNode.GetName()
            if name.startswith("Model_"):
                models.append(dataNode)
    return models

modelNodesToExport = {}

for folderName in folderNamesOfInterest:
    for modelNode in getModelNodesInFolder(folderName):
        modelNodesToExport[modelNode.GetName()] = modelNode

for modelName in extraModelNames:
    try:
        node = slicer.util.getNode(modelName)
        modelNodesToExport[node.GetName()] = node
    except slicer.util.MRMLNodeNotFoundException:
        print(f"[WARN] Model node '{modelName}' not found, skipping.")

print(f"Found {len(modelNodesToExport)} model nodes to export.")

segLogic = slicer.modules.segmentations.logic()

for modelName, modelNode in modelNodesToExport.items():
    print(f"\n=== Processing model: {modelName} ===")

    segNodeName = modelName + "_seg"
    segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", segNodeName)

    segNode.SetReferenceImageGeometryParameterFromVolumeNode(referenceVolumeNode)

    segLogic.ImportModelToSegmentationNode(modelNode, segNode)

    labelName = modelName + "_label"
    labelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", labelName)

    segLogic.ExportVisibleSegmentsToLabelmapNode(segNode, labelNode, referenceVolumeNode)

    outPath = os.path.join(outputDir, modelName + ".nii")
    success = slicer.util.saveNode(labelNode, outPath)

    if success:
        print(f"Saved: {outPath}")
    else:
        print(f"[ERROR] Failed to save NIfTI for {modelName}")

print("\nDone exporting all requested models to NIfTI.")
