#pragma once

#include <vector>
#include <string>
#include <boost/unordered_map.hpp>
#include "StatefulFeatureFunction.h"
#include "FFState.h"
#include "moses/Sentence.h"
#include "moses/TargetPhrase.h"
#include "CoarseLMModel.h"

namespace Moses {

class CoarseBiLMState: public FFState {
	int m_targetLen;
	std::vector<std::string> sourceWords;
	lm::ngram::State lm100State;
	lm::ngram::State lm1600State;
	lm::ngram::State biLMWithoutClusteringState;
	lm::ngram::State biLMWithClusteringState;
public:
	CoarseBiLMState(int targetLen) :m_targetLen(targetLen) {

	}

	CoarseBiLMState(int targetLen, std::vector<std::string> &source_Words,
				lm::ngram::State &lm100_State, lm::ngram::State &lm1600_State, lm::ngram::State &biLMWithoutClustering_State,
				lm::ngram::State &biLMWithClustering_State) : m_targetLen(targetLen), sourceWords(source_Words),
				lm100State(lm100_State), lm1600State(lm1600_State),
				biLMWithoutClusteringState(biLMWithoutClustering_State), biLMWithClusteringState (biLMWithClustering_State){

	}

	int Compare(const FFState& other) const;

	const lm::ngram::State& getBiLmWithClusteringState() const {
		return biLMWithClusteringState;
	}

	const lm::ngram::State& getBiLmWithoutClusteringState() const {
		return biLMWithoutClusteringState;
	}

	const lm::ngram::State& getLm100State() const {
		return lm100State;
	}

	const lm::ngram::State& getLm1600State() const {
		return lm1600State;
	}

	const std::vector<std::string>& getSourceWords() const {
		return sourceWords;
	}
};

class CoarseBiLM: public StatefulFeatureFunction {

protected:
	boost::unordered_map<std::string, std::string> tgtWordToClusterId100;
	boost::unordered_map<std::string, std::string> tgtWordToClusterId1600;
	boost::unordered_map<std::string, std::string> tgtWordToClusterId400;
	boost::unordered_map<std::string, std::string> srcWordToClusterId400;
	boost::unordered_map<std::string, std::string> bitokenToBitokenId;
	boost::unordered_map<std::string, std::string> bitokenIdToClusterId;
	int nGramOrder;
	std::string m_lmPath1600;
	std::string m_lmPath100;
	std::string m_bilmPathWithoutClustering;
	std::string m_bilmPathWithClustering;

public:
	LM* CoarseLM100;
	LM* CoarseLM1600;
	LM* CoarseBiLMWithoutClustering;
	LM* CoarseBiLMWithClustering;

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

private:
	void LoadManyToOneMap(const std::string& path, boost::unordered_map<std::string, std::string> &manyToOneMap);

	void getTargetWords(const Hypothesis& cur_hypo, std::vector<std::string> &targetWords, std::vector<std::string> &targetWords100, std::vector<std::string> &targetWords1600, std::vector<std::string> &targetWords400, boost::unordered_map<int, std::vector<int> > &alignments) const;

	void getSourceWords(const Sentence &sourceSentence, std::vector<std::string> &sourceWords) const;

	void createBitokens(const std::vector<std::string> &sourceWords, const std::vector<std::string> &targetWords, const boost::unordered_map<int, std::vector<int> > &alignments, std::vector<std::string> &bitokenBitokenIDs, std::vector<std::string> &bitokenWordIDs) const;

	size_t getState(const std::vector<std::string> &wordsToScore) const;

	void printList(const std::vector<std::string> &listToPrint) const;

	float getLMScore(const std::vector<std::string> &wordsToScore, const LM* languageModel, lm::ngram::State &state) const;

	std::string getStringFromList(const std::vector<std::string> &listToConvert) const;

	std::string getClusterID(const std::string &key, const boost::unordered_map<std::string, std::string> &clusterIdMap) const;
};

}
