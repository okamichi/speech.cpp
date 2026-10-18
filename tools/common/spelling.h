#pragma once

#include <string>
#include <vector>

// What a word that names nothing the command line knows was likely meant to be, among the words it knows: a flag, a
// subcommand or a model's name with a slip of the keys in it.

/**
 * The candidate nearest `word` by the edits that turn one into the other (a letter added, removed, changed, or two
 * swapped), where it is near enough to be a slip: one edit in a word of up to four letters, two in a longer one. ""
 * where none is, or where two are equally near.
 */
std::string nearest(const std::string & word, const std::vector<std::string> & candidates);
