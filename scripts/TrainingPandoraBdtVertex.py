#!/usr/bin/env python
# example.py

from PandoraBDT import *

if __name__=="__main__":

    # Settings ------------------------------------------------------------------------------------

    trainingFiles     = [
                            '/Users/stevengreen/Documents/PostDoc/LAr/2019/April/VertexBDT/Training/Samples/VertexTraining_ProtoDUNE_BDTVertexing_BEAM_PARTICLE_E_1GeV.txt',
                            '/Users/stevengreen/Documents/PostDoc/LAr/2019/April/VertexBDT/Training/Samples/VertexTraining_ProtoDUNE_BDTVertexing_BEAM_PARTICLE_E_3GeV.txt',
                            '/Users/stevengreen/Documents/PostDoc/LAr/2019/April/VertexBDT/Training/Samples/VertexTraining_ProtoDUNE_BDTVertexing_BEAM_PARTICLE_E_5GeV.txt',
                            '/Users/stevengreen/Documents/PostDoc/LAr/2019/April/VertexBDT/Training/Samples/VertexTraining_ProtoDUNE_BDTVertexing_BEAM_PARTICLE_E_7GeV.txt',
                            '/Users/stevengreen/Documents/PostDoc/LAr/2019/April/VertexBDT/Training/Samples/VertexTraining_ProtoDUNE_BDTVertexing_BEAM_PARTICLE_PI_PLUS_1GeV.txt',
                            '/Users/stevengreen/Documents/PostDoc/LAr/2019/April/VertexBDT/Training/Samples/VertexTraining_ProtoDUNE_BDTVertexing_BEAM_PARTICLE_PI_PLUS_3GeV.txt',
                            '/Users/stevengreen/Documents/PostDoc/LAr/2019/April/VertexBDT/Training/Samples/VertexTraining_ProtoDUNE_BDTVertexing_BEAM_PARTICLE_PI_PLUS_5GeV.txt',
                            '/Users/stevengreen/Documents/PostDoc/LAr/2019/April/VertexBDT/Training/Samples/VertexTraining_ProtoDUNE_BDTVertexing_BEAM_PARTICLE_PI_PLUS_7GeV.txt'
                        ]
    trainingDescription  = '1-7 GeV Electrons and Pions'
    testingFiles      = [
                            {'File' : '/Users/stevengreen/Documents/PostDoc/LAr/2019/April/VertexBDT/Training/Samples/VertexTraining_ProtoDUNE_BDTVertexing_BEAM_PARTICLE_E_1GeV.txt', 'Description': '1 GeV Electrons' },
                            {'File' : '/Users/stevengreen/Documents/PostDoc/LAr/2019/April/VertexBDT/Training/Samples/VertexTraining_ProtoDUNE_BDTVertexing_BEAM_PARTICLE_E_3GeV.txt', 'Description': '3 GeV Electrons' },
			    {'File' : '/Users/stevengreen/Documents/PostDoc/LAr/2019/April/VertexBDT/Training/Samples/VertexTraining_ProtoDUNE_BDTVertexing_BEAM_PARTICLE_E_5GeV.txt', 'Description': '5 GeV Electrons' },
                            {'File' : '/Users/stevengreen/Documents/PostDoc/LAr/2019/April/VertexBDT/Training/Samples/VertexTraining_ProtoDUNE_BDTVertexing_BEAM_PARTICLE_E_7GeV.txt', 'Description': '7 GeV Electrons' },
                            {'File' : '/Users/stevengreen/Documents/PostDoc/LAr/2019/April/VertexBDT/Training/Samples/VertexTraining_ProtoDUNE_BDTVertexing_BEAM_PARTICLE_PI_PLUS_1GeV.txt', 'Description': '1 GeV Pi Plus' },
                            {'File' : '/Users/stevengreen/Documents/PostDoc/LAr/2019/April/VertexBDT/Training/Samples/VertexTraining_ProtoDUNE_BDTVertexing_BEAM_PARTICLE_PI_PLUS_3GeV.txt', 'Description': '3 GeV Pi Plus' },
                            {'File' : '/Users/stevengreen/Documents/PostDoc/LAr/2019/April/VertexBDT/Training/Samples/VertexTraining_ProtoDUNE_BDTVertexing_BEAM_PARTICLE_PI_PLUS_5GeV.txt', 'Description': '5 GeV Pi Plus' },
                            {'File' : '/Users/stevengreen/Documents/PostDoc/LAr/2019/April/VertexBDT/Training/Samples/VertexTraining_ProtoDUNE_BDTVertexing_BEAM_PARTICLE_PI_PLUS_7GeV.txt', 'Description': '7 GeV Pi Plus' },
                        ]
    bdtName           = 'Vertex_ProtoDUNE'
    treeDepth         = int(sys.argv[1])
    nTrees            = int(sys.argv[2])
    trainTestSplit    = 0.5

    plotFeatures      = False # Draws distributions of signal and background class features, then exits
    serializeToPkl    = True
    serializeToXml    = True
    loadFromPkl       = False
    makeScorePlots    = True # Makes plots of BDT score for training and testing samples
    xmlFileName       = bdtName + '_NTrees_' + str(nTrees) + '_TreeDepth_' + str(treeDepth) + '.xml'
    pklFileName       = bdtName + '_NTrees_' + str(nTrees) + '_TreeDepth_' + str(treeDepth) + '.pkl'

    #----------------------------------------------------------------------------------------------

    if plotFeatures:
        # Load the training data
        OverwriteStdout('Loading training set data for plotting from files ' + ','.join(trainingFiles) + '\n')
        trainSet, nFeatures, nExamplesList = LoadDatasets(trainingFiles, ',')
        X_org, Y_org = SplitTrainingSet(trainSet, nFeatures)

        # Plot Variables then Exit
        DrawVariables(X_org, Y_org)
        Correlation(X_org, Y_org)
        sys.exit()

    if loadFromPkl:
        OverwriteStdout('Loading model from file ' + pklFileName + '\n')
        bdtModel = LoadFromPkl(pklFileName)

    else:
        # Load the training data
        OverwriteStdout('Loading training set data from files ' + ','.join(trainingFiles) + '\n')
        trainSet, nFeatures, nExamplesList = LoadDatasets(trainingFiles, ',')
        X_org, Y_org = SplitTrainingSet(trainSet, nFeatures)

        sample_weights_org = []
        for nExamples in nExamplesList:
            for example in range(nExamples):
                sample_weights_org.append(1/nExamples)
        sample_weights_org = np.asarray(sample_weights_org)

        # Train the BDT
        X, Y, sample_weights = Randomize(X_org, Y_org, sample_weights=sample_weights_org)
        X_train, Y_train, sample_weights_train, X_test, Y_test, sample_weights_test = Sample(X, Y, sample_weights, trainTestSplit)

        OverwriteStdout('Training AdaBoostClassifer...')
        bdtModel, trainingTime = TrainAdaBoostClassifer(X_train, Y_train, n_estimatorsValue=nTrees, max_depthValue=treeDepth, sample_weights=sample_weights_train)

        OverwriteStdout(('Trained AdaBoostClassifer with ' + str(nFeatures) + ' features and ' + str(sum(nExamplesList) * trainTestSplit) + ' examples (%d seconds, %i TreeDepth, %i nTrees)\n' % (trainingTime, treeDepth, nTrees)))

        # Validate the model
        modelScore = ValidateModel(bdtModel, X_test, Y_test)
        OverwriteStdout('Model score: %.2f%%\n' % (modelScore * 100))

        if serializeToXml:
            OverwriteStdout('Writing model to xml file ' + xmlFileName + '\n')
            datetimeString = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            WriteXmlFile(xmlFileName, bdtModel)

        if serializeToPkl:
            OverwriteStdout('Writing model to pkl file ' + pklFileName + '\n')
            SerializeToPkl(pklFileName, bdtModel)

    # Do other stuff with your trained/loaded model
    # ...

    if makeScorePlots:
        parameters = {
                         'ClassNames':['Target Vertices','Remaining Vertices'],
                         'SignalDefinition': [1, 0],
                         'PlotColors': ['b', 'r'],
                         'nBins': 100,
                         'PlotStep': 1.0,
                         'OptimalBinCut': 0,
                         'OptimalScoreCut': 0.0,
                         'nTrees': nTrees,
                         'TreeDepth': treeDepth
                     }

        FindOptimalSignificanceCut(bdtModel, X_train, Y_train, parameters)
        PlotBdtScores(bdtModel, X_test, Y_test, trainingDescription, parameters)

        for idx, testFile in enumerate(testingFiles):
            testSet, nFeaturesTest, nExamplesTest = LoadData(testFile['File'], ',')
            X_test_data, Y_test_data = SplitTrainingSet(testSet, nFeaturesTest)
            PlotBdtScores(bdtModel, X_test_data, Y_test_data, testFile['Description'], parameters)

