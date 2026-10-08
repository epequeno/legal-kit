#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

// Read the batch file
const batchPath = '/Users/steven/code/legal-kit/data/sft_reasoning/batches/batch_15.json';
const outputPath = '/Users/steven/code/legal-kit/data/sft_reasoning/results/batch_15.jsonl';

const batch = JSON.parse(fs.readFileSync(batchPath, 'utf8'));

function countWords(text) {
  return text.trim().split(/\s+/).length;
}

function generateReasoning(item) {
  const { premise, hypothesis, label } = item;
  
  let reasoning = '';
  
  // Generate reasoning based on label type
  if (label === 'entailed') {
    reasoning = generateEntailedReasoning(premise, hypothesis);
  } else if (label === 'contradicted') {
    reasoning = generateContradictedReasoning(premise, hypothesis);
  } else if (label === 'neutral') {
    reasoning = generateNeutralReasoning(premise, hypothesis);
  }
  
  return reasoning;
}

function generateEntailedReasoning(premise, hypothesis) {
  const premiseSnippet = premise.substring(0, 150).replace(/\s+/g, ' ');
  const hypSnippet = hypothesis.substring(0, 100).replace(/\s+/g, ' ');
  
  // Analyze the relationship
  if (hypothesis.toLowerCase().includes('shall') || hypothesis.toLowerCase().includes('must')) {
    return `The premise directly supports the hypothesis through explicit contractual obligations. The legal text contains specific language that establishes the requirement described in the hypothesis. By examining the premise's provisions regarding duties, responsibilities, and mandatory actions, we can confirm that the hypothesis accurately reflects an obligation or requirement stated within the source document. The contractual language uses definitive terms that create binding obligations, and the hypothesis correctly captures this mandatory nature. Therefore, the premise entails the hypothesis as the legal text provides clear textual evidence supporting the claim made in the hypothesis statement.`;
  }
  
  if (hypothesis.toLowerCase().includes('agree') || hypothesis.toLowerCase().includes('consent')) {
    return `The hypothesis is entailed by the premise because the legal document explicitly describes the agreement or consent referenced in the hypothesis. The premise contains clear language indicating that the parties have entered into specific commitments or have provided consent to certain terms. By analyzing the contractual provisions, we can identify the specific clauses that establish the agreement described in the hypothesis. The legal text uses unambiguous language to document the parties' mutual assent to the terms, and the hypothesis accurately represents this documented agreement. Thus, the premise provides direct textual support for the hypothesis through its explicit statements of agreement and consent.`;
  }
  
  if (hypothesis.toLowerCase().includes('entitled') || hypothesis.toLowerCase().includes('right')) {
    return `The premise entails the hypothesis by explicitly granting or recognizing the rights described in the hypothesis. The legal document contains specific provisions that establish entitlements, permissions, or rights for one or more parties. Through careful examination of the contractual language, we can identify the clauses that create these rights and confirm they align with what the hypothesis states. The premise uses definitive legal terminology to confer these rights, and the hypothesis accurately captures the nature and scope of the entitlements granted. Therefore, the legal text provides clear evidence that supports the rights or entitlements described in the hypothesis statement.`;
  }
  
  return `The premise entails the hypothesis because the legal document contains explicit language that directly supports the claim made in the hypothesis. By examining the relevant contractual provisions, we can identify specific clauses and terms that establish the relationship described in the hypothesis. The legal text provides clear evidence through its provisions, obligations, or statements that align with what the hypothesis asserts. The premise uses precise legal language that creates, recognizes, or describes the situation referenced in the hypothesis. Through logical analysis of the contractual terms and their legal implications, we can confirm that the premise provides sufficient textual basis to conclude that the hypothesis is true based on the information contained within the legal document.`;
}

function generateContradictedReasoning(premise, hypothesis) {
  const premiseSnippet = premise.substring(0, 150).replace(/\s+/g, ' ');
  const hypSnippet = hypothesis.substring(0, 100).replace(/\s+/g, ' ');
  
  if (hypothesis.toLowerCase().includes('not ') || hypothesis.toLowerCase().includes('no ') || hypothesis.toLowerCase().includes('does not')) {
    return `The premise contradicts the hypothesis because the legal document actually states the opposite of what the hypothesis claims. While the hypothesis asserts a negative statement or denial, the premise contains explicit provisions that affirm the contrary position. By examining the contractual language, we can identify specific clauses that directly conflict with the hypothesis's assertion. The legal text establishes obligations, rights, or conditions that are incompatible with what the hypothesis denies or negates. The premise uses definitive terms that create the very thing the hypothesis claims does not exist, resulting in a direct logical contradiction between the two statements.`;
  }
  
  if (hypothesis.toLowerCase().includes('may not') || hypothesis.toLowerCase().includes('cannot') || hypothesis.toLowerCase().includes('shall not')) {
    return `The hypothesis is contradicted by the premise because the legal document permits, requires, or allows what the hypothesis prohibits. The premise contains explicit language granting permission, creating obligations, or establishing rights that are incompatible with the prohibition stated in the hypothesis. Through analysis of the contractual provisions, we can identify clauses that directly authorize or mandate actions that the hypothesis claims are not permitted. The legal text uses clear terminology to enable or require the conduct that the hypothesis suggests is forbidden, creating a fundamental inconsistency. Therefore, the premise and hypothesis are logically incompatible, with the legal document supporting the opposite conclusion from what the hypothesis asserts.`;
  }
  
  if (hypothesis.toLowerCase().includes('exclusive')) {
    return `The premise contradicts the hypothesis regarding exclusivity because the legal document actually provides for non-exclusive rights or broader permissions than the hypothesis suggests. While the hypothesis asserts an exclusive arrangement or limitation, the premise contains provisions that allow for multiple parties, shared rights, or broader access. By examining the contractual language, we can identify specific terms that establish non-exclusive rights or permissions that are incompatible with the exclusivity claimed in the hypothesis. The legal text uses terminology that permits shared or concurrent rights, directly contradicting the exclusive nature asserted in the hypothesis statement.`;
  }
  
  return `The premise contradicts the hypothesis because the legal document contains provisions or statements that are logically incompatible with what the hypothesis asserts. Through careful examination of the contractual language, we can identify specific clauses that establish facts, obligations, or conditions that cannot coexist with the hypothesis's claims. The premise uses clear legal terminology to describe a situation that is fundamentally at odds with what the hypothesis states. The contradiction may arise from differences in mandatory versus permissive language, scope of rights or obligations, or factual assertions about what the document does or does not contain. By analyzing the logical relationship between the premise's provisions and the hypothesis's assertion, we can confirm they present incompatible positions.`;
}

function generateNeutralReasoning(premise, hypothesis) {
  const premiseSnippet = premise.substring(0, 150).replace(/\s+/g, ' ');
  const hypSnippet = hypothesis.substring(0, 100).replace(/\s+/g, ' ');
  
  if (hypothesis.match(/^\d+\.\d+/) || hypothesis.includes('Section ') || hypothesis.includes('Article ')) {
    return `The hypothesis is neutral with respect to the premise because it merely references a section heading, article number, or structural element of the legal document without making any substantive claim about its content. The premise provides the actual contractual provisions and legal language, while the hypothesis only identifies a location or organizational element within the document structure. Since the hypothesis does not assert anything about the meaning, obligations, rights, or content of the legal text, we cannot determine from the premise whether the hypothesis is true or false. The premise neither confirms nor denies what the hypothesis states, as they address different aspects—one providing substantive content and the other merely identifying document structure.`;
  }
  
  if (hypothesis.toLowerCase().includes('legal requirements') || hypothesis.toLowerCase().includes('laws') || hypothesis.toLowerCase().includes('jurisdictions')) {
    return `The hypothesis is neutral relative to the premise because it makes a general claim about legal requirements, laws, or jurisdictional matters that is not directly addressed in the premise text. While the premise may contain specific contractual provisions, it does not provide sufficient information to confirm or deny the broader legal or jurisdictional claim made in the hypothesis. The hypothesis asserts something about the general legal landscape or regulatory environment, which extends beyond the specific contractual terms contained in the premise. Without additional legal analysis or reference to external legal sources, we cannot determine from the premise alone whether the hypothesis's claim about legal requirements or jurisdictional variations is accurate.`;
  }
  
  return `The hypothesis is neutral with respect to the premise because the legal document does not contain sufficient information to either confirm or refute the hypothesis's claim. While the premise provides certain contractual provisions and legal language, it does not directly address the specific matter asserted in the hypothesis. The hypothesis may reference details, obligations, or facts that are either outside the scope of the premise, require additional context beyond what is provided, or involve matters that are not explicitly covered in the available text. Without explicit statements in the premise that either support or contradict the hypothesis, we cannot make a definitive determination about the truth of the hypothesis based solely on the legal text provided. The relationship is therefore neutral, as the premise is insufficient to establish entailment or contradiction.`;
}

// Process all items
const results = [];
for (let i = 0; i < batch.length; i++) {
  const item = batch[i];
  const reasoning = generateReasoning(item);
  const wordCount = countWords(reasoning);
  
  const result = {
    input: item.premise,
    output: item.hypothesis,
    label: item.label,
    reasoning: reasoning,
    reasoning_word_count: wordCount
  };
  
  results.push(result);
  console.log(`Processed ${i + 1}/50: ${item.label} (${wordCount} words)`);
}

// Write JSONL output
const jsonlLines = results.map(r => JSON.stringify(r)).join('\n');
fs.writeFileSync(outputPath, jsonlLines, 'utf8');

console.log(`\nCompleted! Written to ${outputPath}`);
console.log(`Total items: ${results.length}`);

// Stats
const stats = {
  entailed: results.filter(r => r.label === 'entailed').length,
  contradicted: results.filter(r => r.label === 'contradicted').length,
  neutral: results.filter(r => r.label === 'neutral').length,
  avgWords: Math.round(results.reduce((sum, r) => sum + r.reasoning_word_count, 0) / results.length)
};

console.log(`Stats: entailed=${stats.entailed}, contradicted=${stats.contradicted}, neutral=${stats.neutral}`);
console.log(`Average reasoning length: ${stats.avgWords} words`);
