def search_basic(stuff, organizations):
    results = []
    for organization in organizations:
        if stuff in organization.all_data().lower():
            results.append(organization)
    return results


if __name__ == "__main__":
    search_basic()
