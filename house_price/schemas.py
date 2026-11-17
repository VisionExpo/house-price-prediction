"""Validated request schemas for the prediction API."""

from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    gr_liv_area: int = Field(ge=334, le=5642, alias="GrLivArea")
    lot_area: int = Field(ge=1300, le=215245, alias="LotArea")
    overall_qual: int = Field(ge=1, le=10, alias="OverallQual")
    year_built: int = Field(ge=1872, le=2010, alias="YearBuilt")
    bedrooms: int = Field(ge=0, le=8, alias="BedroomAbvGr")
    full_baths: int = Field(ge=0, le=3, alias="FullBath")
    garage_cars: int = Field(ge=0, le=4, alias="GarageCars")
    neighborhood: str = Field(min_length=1, max_length=20, alias="Neighborhood")

    model_config = {"populate_by_name": True}

    def model_features(self) -> dict[str, int | str]:
        return {
            "GrLivArea": self.gr_liv_area,
            "LotArea": self.lot_area,
            "OverallQual": self.overall_qual,
            "YearBuilt": self.year_built,
            "BedroomAbvGr": self.bedrooms,
            "FullBath": self.full_baths,
            "GarageCars": self.garage_cars,
            "Neighborhood": self.neighborhood,
        }
