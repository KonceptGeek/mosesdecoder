#pragma once

#include <vector>
#include <string>
#include "StatefulFeatureFunction.h"
#include "FFState.h"
#include "moses/Sentence.h"
#include "moses/TargetPhrase.h"
#include "CoarseLMModel.h"

namespace Moses {

class CoarseBiLMState: public FFState {
	int m_targetLen;
public:
	CoarseBiLMState(int targetLen) :
			m_targetLen(targetLen) {
	}

	int Compare(const FFState& other) const;
};

class CoarseBiLM: public StatefulFeatureFunction {

protected:
	std::map<std::string, std::string> tgtWordToClusterId;
	std::map<std::string, std::string> srcWordToClusterId;
	bool cvtSrcToClusterId;
	std::map<std::string, std::string> bitokenToBitokenId;
	bool cvtBitokenToBitokenId;
	std::map<std::string, std::string> bitokenIdToClusterId;
	bool cvtBitokenIdToClusterId;
	int nGramOrder;
	std::string m_lmPath;

public:
	LM* CoarseLM;

	CoarseBiLM(const std::string &line);
	~CoarseBiLM();

	void Load();

	bool IsUseable(const FactorMask &mask) const {
		return true;
	}
	virtual const FFState* EmptyHypothesisState(const InputType &input) const {
		return new CoarseBiLMState(0);
	}

	void EvaluateInIsolation(const Phrase &source,
			const TargetPhrase &targetPhrase,
			ScoreComponentCollection &scoreBreakdown,
			ScoreComponentCollection &estimatedFutureScore) const;
	void EvaluateWithSourceContext(const InputType &input,
			const InputPath &inputPath, const TargetPhrase &targetPhrase,
			const StackVec *stackVec, ScoreComponentCollection &scoreBreakdown,
			ScoreComponentCollection *estimatedFutureScore = NULL) const;

	void EvaluateTranslationOptionListWithSourceContext(const InputType &input,
			const TranslationOptionList &translationOptionList) const;

	FFState* EvaluateWhenApplied(const Hypothesis& cur_hypo,
			const FFState* prev_state,
			ScoreComponentCollection* accumulator) const;
	FFState* EvaluateWhenApplied(const ChartHypothesis& /* cur_hypo */,
			int /* featureID - used to index the state in the previous hypotheses */,
			ScoreComponentCollection* accumulator) const;

	void SetParameter(const std::string& key, const std::string& value);

	void readLanguageModel(const char *);

private:
	void LoadManyToOneMap(const std::string& path, std::map<std::string, std::string> &manyToOneMap);

	void getTargetWords(const Hypothesis& cur_hypo, std::vector<std::string> &targetWords, std::map<int, std::vector<int> > &alignments) const;

	void getPreviousTargetWords(const Hypothesis& cur_hypo, int previousWordsNeeded, std::vector<std::string> &targetWords, std::map<int, std::vector<int> > &alignments) const;

	void getSourceWords(const Sentence &sourceSentence, std::vector<std::string> &sourceWords) const;

	void replaceWordsWithClusterID(const std::vector<std::string> &words, const std::map<std::string, std::string> &clusterIdMap, std::vector<std::string> &wordClusterIDs) const;

	void createBitokens(const std::vector<std::string> &sourceWords, const std::vector<std::string> &targetWords, const std::map<int, std::vector<int> > &alignments, std::vector<std::string> &bitokens) const;

	size_t getState(const Hypothesis& cur_hypo) const;

	void printList(const std::vector<std::string> &listToPrint) const;

  std::string getStringFromList(const std::vector<std::string> &listToConvert) const;
};

}
